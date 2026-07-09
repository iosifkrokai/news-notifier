"""Active cancellation of a market's in-flight/queued jobs.

Unsubscribing only flips a market to `paused`, and every process_market /
process_candidate / deliver_batch job already guards on that status and no-ops
when it runs. But a candidate job that's *already mid-flight* (e.g. blocked on a
multi-minute local-LLM extraction) was started before the status flipped and
keeps burning CPU to completion, and queued candidates still spin up a DB session
each just to bail. To actually stop spending resources on a market nobody is
watching anymore, process_market records the ids of the candidate jobs it fans
out (register_market_job) and unsubscribe aborts them (abort_market_jobs) via
arq's abort mechanism — see WorkerSettings.allow_abort_jobs.

Kept dependency-light (arq + the redis pool only, no app.worker.tasks import) so
the API layer can abort without pulling in playwright/fastembed.
"""

from arq.constants import abort_jobs_ss
from arq.utils import timestamp_ms

# Backstop TTL on the per-market id set, refreshed on every add. It only needs to
# outlive a market's in-flight job backlog; stale ids that outlive their jobs are
# harmless (aborting an already-finished/unknown job_id is a no-op), the TTL just
# keeps the set from growing without bound for a long-lived active market.
_JOB_SET_TTL_SECONDS = 24 * 60 * 60


def _market_jobs_key(market_id: str) -> str:
    return f"arq:market-jobs:{market_id}"


async def register_market_job(redis, market_id: str, job_id: str) -> None:
    key = _market_jobs_key(market_id)
    await redis.sadd(key, job_id)
    await redis.expire(key, _JOB_SET_TTL_SECONDS)


async def abort_market_jobs(redis, market_id: str) -> int:
    """Signal every tracked job for this market for abort — a worker running with
    allow_abort_jobs=True cancels the ones currently executing and skips the ones
    still queued — then drop the tracking set. Returns how many ids were signalled.
    Writing straight to arq's abort sorted set (rather than Job.abort, which then
    blocks polling for each job's result) keeps this a fast fire-and-forget call
    safe to await inside an API request handler."""
    key = _market_jobs_key(market_id)
    raw = await redis.smembers(key)
    job_ids = [j.decode() if isinstance(j, (bytes, bytearray)) else j for j in raw]
    if job_ids:
        now = timestamp_ms()
        await redis.zadd(abort_jobs_ss, {job_id: now for job_id in job_ids})
    await redis.delete(key)
    return len(job_ids)
