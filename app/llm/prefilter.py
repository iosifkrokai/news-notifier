"""Cheap semantic pre-filter that runs *before* the expensive per-candidate
pipeline (scrape + local-LLM extraction + embed). Search fans every query out to
every source, so a single cycle can surface dozens of fresh URLs, most only
loosely related to the market. Ranking their titles against the market
description with the local FastEmbed model costs milliseconds per title and lets
us spend the slow work only on the strongest candidates — see the wiring in
app.worker.tasks.process_market and the thresholds in Settings.
"""

import numpy as np

from app.llm.embeddings import embed_texts


def _cosine(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of `query` against each row of `matrix`. Normalizes
    explicitly rather than assuming the embedding model returns unit vectors, so
    this stays correct regardless of the configured embedding_model."""
    query_norm = np.linalg.norm(query)
    row_norms = np.linalg.norm(matrix, axis=1)
    # Guard the degenerate all-zero vector (e.g. an empty title) -> similarity 0.
    denom = np.where(row_norms == 0, 1.0, row_norms) * (query_norm or 1.0)
    return (matrix @ query) / denom


def rank_by_similarity(
    query_vec: list[float],
    candidate_vecs: list[list[float]],
    candidates: list[dict],
    top_k: int,
    min_similarity: float,
) -> list[dict]:
    """Pure selection step (no I/O, no model load — unit-tested directly): keep
    the candidates whose title embedding is at least `min_similarity` to the
    market description, highest first, capped at `top_k`. `top_k <= 0` disables
    the filter and returns the candidates unchanged."""
    if top_k <= 0 or not candidates:
        return candidates

    sims = _cosine(np.asarray(query_vec, dtype=np.float32), np.asarray(candidate_vecs, dtype=np.float32))
    ranked = sorted(
        (
            (float(sim), candidate)
            for sim, candidate in zip(sims, candidates)
            if sim >= min_similarity
        ),
        key=lambda pair: pair[0],
        reverse=True,
    )
    return [candidate for _, candidate in ranked[:top_k]]


async def select_relevant_candidates(
    market_description: str,
    candidates: list[dict],
    top_k: int,
    min_similarity: float,
) -> list[dict]:
    """Embed the market description and every candidate title in one batch, then
    delegate to rank_by_similarity. Returns candidates unchanged when the filter
    is disabled (top_k <= 0) or there's nothing to rank."""
    if top_k <= 0 or not candidates:
        return candidates

    texts = [market_description] + [(c.get("title") or "") for c in candidates]
    vectors = await embed_texts(texts)
    query_vec, candidate_vecs = vectors[0], vectors[1:]
    return rank_by_similarity(query_vec, candidate_vecs, candidates, top_k, min_similarity)
