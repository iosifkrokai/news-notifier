import asyncio
import functools
from collections.abc import Awaitable, Callable
from typing import TypeVar

from arq.worker import Retry

T = TypeVar("T")


def serializable_job_errors(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """Ensure a failed arq job's result is deserializable by consumers that only
    have arq + the stdlib (notably the arq-ui monitor container, which does NOT
    ship our deps).

    arq pickles a job's result — including the exception on failure — into Redis.
    If that exception's type comes from a third-party lib (httpx, sqlalchemy,
    playwright, fastembed, ...), any reader without that lib installed blows up
    with `ModuleNotFoundError` while unpickling. We re-raise those as a plain
    RuntimeError (a builtin, always importable) carrying the original type name
    and message. `from exc` keeps the full chained traceback in the worker's own
    log; __cause__ is not pickled, so the stored result stays clean and portable.

    Builtin exceptions (e.g. the intentional RuntimeError deliver_batch raises to
    trigger an arq retry) are already deserializable everywhere, so they pass
    through untouched — no redundant double-wrapping. Retry (arq's own
    control-flow signal) and CancelledError likewise pass through."""

    @functools.wraps(func)
    async def wrapper(*args: object, **kwargs: object) -> T:
        try:
            return await func(*args, **kwargs)
        except (Retry, asyncio.CancelledError):
            raise
        except Exception as exc:
            if type(exc).__module__ == "builtins":
                raise
            raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc

    return wrapper
