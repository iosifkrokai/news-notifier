import asyncio
from functools import lru_cache

from fastembed import TextEmbedding

from app.config import get_settings


@lru_cache
def _model() -> TextEmbedding:
    return TextEmbedding(model_name=get_settings().embedding_model)


async def embed_text(text: str) -> list[float]:
    return await asyncio.to_thread(_embed_sync, text[:4000])


def _embed_sync(text: str) -> list[float]:
    return next(_model().embed([text])).tolist()


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch variant of embed_text — one FastEmbed call for many short strings,
    used by the candidate pre-filter (app.llm.prefilter) to rank a whole cycle's
    candidate titles against the market description in a single pass."""
    return await asyncio.to_thread(_embed_many_sync, [t[:4000] for t in texts])


def _embed_many_sync(texts: list[str]) -> list[list[float]]:
    return [vec.tolist() for vec in _model().embed(texts)]
