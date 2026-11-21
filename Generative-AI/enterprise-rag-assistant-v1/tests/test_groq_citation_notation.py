"""Offline contract tests: normalize notation, never fabricate source references."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.generation.citation_builder import CitationBuilder
from src.generation.citation_normalizer import normalize_explicit_citations
from src.generation.failover_generator import GroqAnswerGenerator


def sources(count=3):
    return [
        {"evidence_id": i, "chunk_id": f"DOC-{i}", "document_id": "DOC", "page_start": i}
        for i in range(1, count + 1)
    ]


@pytest.mark.parametrize(("raw", "normalized"), [
    ("Supported. [Evidence 1]", "Supported. [Evidence 1]"),
    ("Supported. [Evidence 1, Evidence 2]", "Supported. [Evidence 1, Evidence 2]"),
    ("Supported. [Evidence 1, 2]", "Supported. [Evidence 1, Evidence 2]"),
    ("Supported. [Evidence 1 and 2]", "Supported. [Evidence 1, Evidence 2]"),
    ("Supported. [Evidence 1; Evidence 3]", "Supported. [Evidence 1, Evidence 3]"),
    ("Supported. [Evidence 1 & Evidence 3]", "Supported. [Evidence 1, Evidence 3]"),
    ("Supported. [Evidence #1, #2]", "Supported. [Evidence 1, Evidence 2]"),
    ("Supported. (Evidence 2)", "Supported. [Evidence 2]"),
    ("Supported. 【Evidence 2】", "Supported. [Evidence 2]"),
    ("Supported. Evidence 2.", "Supported. [Evidence 2]."),
    ("Supported. **Evidence 2**", "Supported. **[Evidence 2]**"),
    ("Supported. [Evidence\u202f2]", "Supported. [Evidence 2]"),
    ("Supported. Evidence 2 and Evidence 3.",
     "Supported. [Evidence 2] and [Evidence 3]."),
])
def test_explicit_citation_notation_is_normalized(raw, normalized):
    actual = normalize_explicit_citations(raw)
    assert actual == normalized
    outcome = CitationBuilder().build(actual, sources())
    assert outcome["citation_count"] >= 1
    assert outcome["answer"] == normalized


@pytest.mark.parametrize("raw", [
    "Unsupported citation [Evidence 99]",
    "Unsupported citation Evidence 99",
    "Unsupported citation [Evidence 1-3]",
    "Unsupported citation [Evidence 1–3]",
    "Unsupported citation [Evidence 1, 99]!",
    "Unsupported citation [Evidence 1, Evidence 99]!",
])
def test_normalization_does_not_make_unknown_or_ambiguous_citations_valid(raw):
    actual = normalize_explicit_citations(raw)
    with pytest.raises(ValueError):
        CitationBuilder().build(actual, sources())


def test_unlabelled_numbers_are_not_converted_to_evidence_references():
    assert normalize_explicit_citations("Policy [1], page 17, version 2.0.") == (
        "Policy [1], page 17, version 2.0."
    )


def test_malformed_bracket_group_is_not_partially_repaired():
    raw = "Contradictory [Evidence 1-3] and extra [Evidence 2, 3-4]."
    assert normalize_explicit_citations(raw) == raw
    with pytest.raises(ValueError, match="Malformed"):
        CitationBuilder().build(raw, sources())


def test_groq_normalizes_api_response_before_authoritative_citation_check(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "offline-placeholder")
    monkeypatch.delenv("GROQ_FALLBACK_MODEL", raising=False)

    def fake_post(*_args, **_kwargs):
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "choices": [{
                    "message": {"content": "Approved by IAM. [Evidence 1, 2]"},
                    "finish_reason": "stop",
                }],
                "usage": {},
            },
        )

    monkeypatch.setattr("src.generation.failover_generator.requests.post", fake_post)
    answer = GroqAnswerGenerator(system_prompt="Use evidence").generate(
        "What does IAM require?", "[Evidence 1] A. [Evidence 2] B."
    )
    assert answer["model"] == "openai/gpt-oss-120b"
    assert answer["answer"] == "Approved by IAM. [Evidence 1, Evidence 2]"
    citations = CitationBuilder().build(answer["answer"], sources())
    assert [c["evidence_id"] for c in citations["citations"]] == [1, 2]
