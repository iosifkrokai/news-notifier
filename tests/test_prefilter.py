from app.llm.prefilter import rank_by_similarity


def _c(title: str) -> dict:
    return {"title": title, "url": f"https://example.com/{title.replace(' ', '-')}"}


def test_ranks_by_cosine_and_caps_at_top_k():
    # query points along axis 0; candidates get progressively less aligned.
    query = [1.0, 0.0]
    candidates = [_c("a"), _c("b"), _c("c")]
    vecs = [[1.0, 0.0], [0.9, 0.4], [0.1, 1.0]]  # sims ~ 1.0, 0.91, 0.10
    result = rank_by_similarity(query, vecs, candidates, top_k=2, min_similarity=0.0)
    assert [c["title"] for c in result] == ["a", "b"]  # closest two, best first


def test_min_similarity_drops_weak_candidates():
    query = [1.0, 0.0]
    candidates = [_c("strong"), _c("weak")]
    vecs = [[1.0, 0.0], [0.0, 1.0]]  # sims 1.0 and 0.0
    result = rank_by_similarity(query, vecs, candidates, top_k=10, min_similarity=0.5)
    assert [c["title"] for c in result] == ["strong"]


def test_top_k_zero_disables_filter_and_returns_all():
    candidates = [_c("a"), _c("b")]
    result = rank_by_similarity([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], candidates, top_k=0, min_similarity=0.9)
    assert result == candidates


def test_zero_vector_title_gets_zero_similarity_not_error():
    # An empty title embeds to something the model returns; guard the degenerate
    # all-zero case so it scores 0 rather than dividing by zero.
    query = [1.0, 0.0]
    candidates = [_c("real"), _c("empty")]
    vecs = [[1.0, 0.0], [0.0, 0.0]]
    result = rank_by_similarity(query, vecs, candidates, top_k=10, min_similarity=0.5)
    assert [c["title"] for c in result] == ["real"]


def test_empty_candidates_returns_empty():
    assert rank_by_similarity([1.0, 0.0], [], [], top_k=5, min_similarity=0.1) == []
