from arq import cron
from arq.connections import RedisSettings

from app.config import get_settings
from app.worker.scheduler import enqueue_due_markets, enqueue_stuck_deliveries
from app.worker.tasks import deliver_batch, process_candidate, process_market

_settings = get_settings()


class WorkerSettings:
    """arq worker entrypoint: `arq app.worker.settings.WorkerSettings`"""

    redis_settings = RedisSettings.from_dsn(_settings.redis_url)
    functions = [process_market, process_candidate, deliver_batch]
    # enqueue_due_markets is now just a safety net (process_market self-schedules
    # its own next run — see app.worker.tasks.process_market), so it only needs
    # to run often enough to catch a lost job within OVERDUE_THRESHOLD, not every
    # minute. enqueue_stuck_deliveries stays on its original cadence — it's still
    # the primary recovery path for stuck deliveries, not a backstop for one.
    cron_jobs = [
        cron(enqueue_due_markets, minute={0, 10, 20, 30, 40, 50}),
        cron(enqueue_stuck_deliveries, minute=set(range(60))),
    ]
    max_tries = 6
    # No single job is heavy anymore: process_market only dispatches, and each
    # process_candidate is one scrape + one LLM extraction + one embed. The old
    # 600s existed because the whole per-market batch ran in one job.
    job_timeout = 180
    # process_candidate is the only job that calls the LLM (extract_and_score),
    # one call per job. Keep this close to the `llm` service's --parallel
    # (docker-compose.yml, currently 3): running more process_candidate jobs
    # concurrently than the LLM server has slots for doesn't add throughput,
    # it just queues extra requests on llama.cpp's side while still holding a
    # Playwright browser + FastEmbed call open on ours. +1 over --parallel
    # gives embed_text/scrape_urls-only work (nothing waiting on the LLM slot)
    # a bit of headroom to proceed independently.
    max_jobs = 4
