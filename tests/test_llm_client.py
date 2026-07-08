import pytest

from app.llm.client import _ensure_required_keys
from app.llm.schemas import EXTRACTION_SCHEMA, QUERY_GEN_SCHEMA


def test_passes_when_all_required_keys_present():
    data = {
        "is_relevant": True,
        "title": "t",
        "summary": "s",
        "relevance_score": 0.5,
        "relevance_reasoning": "r",
        "credibility_signal": 0.5,
        "credibility_reasoning": "r",
        "impact_hint": "neutral",
        "proofs": [],
    }
    _ensure_required_keys(data, EXTRACTION_SCHEMA, "extraction")  # must not raise


def test_raises_with_clear_message_when_grammar_constraint_was_not_honored():
    # Simulates llama.cpp's documented fail-open behavior: a 200 response whose
    # body is valid JSON, but not shaped like what the schema required — e.g.
    # the model just answered in prose wrapped in a single JSON string field.
    data = {"answer": "This article is not relevant to the market."}

    with pytest.raises(ValueError, match="missing required field"):
        _ensure_required_keys(data, EXTRACTION_SCHEMA, "extraction")


def test_raises_listing_every_missing_field():
    data = {"queries": ["a", "b"]}
    # QUERY_GEN_SCHEMA only requires "queries", so a well-formed response
    # shouldn't raise even though it doesn't share fields with EXTRACTION_SCHEMA.
    _ensure_required_keys(data, QUERY_GEN_SCHEMA, "query_gen")

    with pytest.raises(ValueError, match=r"\['queries'\]"):
        _ensure_required_keys({}, QUERY_GEN_SCHEMA, "query_gen")
