"""Offline regressions for document-control context selection."""

from types import SimpleNamespace

import pytest

from src.generation.citation_builder import CitationBuilder
from src.rag.answer_composer import compose_answer
from src.retrieval.context_builder import ContextBuilder


def _candidates(document_id: str) -> list[dict]:
    special = {
        11: "Owner: Arjun Malhotra; Approver: Ananya Rao",
        13: "Owner: Ritika Nair; Approver: Kavita Menon",
        16: "Effective date: 2026-04-01",
    }
    candidates = []
    for rank in range(1, 21):
        is_table = rank in special
        candidates.append({
            "rank": rank,
            "chunk": {
                "chunk_id": f"{document_id}-{'TABLE' if is_table else 'BODY'}-C{rank:03d}",
                "document_id": document_id,
                "document_name": "Test policy",
                "text": special.get(rank, f"Unrelated policy section {rank}"),
                "token_count": 30,
            },
        })
    return candidates


def _compose(question: str, candidates: list[dict]):
    seen_context = []

    def generate(**kwargs):
        seen_context.append(kwargs["context"])
        return {"answer": "Answer supported [Evidence 1]", "model": "stub", "usage": {}}

    result = compose_answer(
        question=question,
        candidates=candidates,
        retrieval_mode="hybrid",
        retrieval_source="saved_hybrid_checkpoint",
        retrieval_seconds=None,
        context_builder=ContextBuilder(max_chunks=8, max_context_tokens=4000),
        generator=SimpleNamespace(generate=generate),
        citation_builder=CitationBuilder(),
        evaluation_trace=True,
    )
    return result, seen_context[0]


@pytest.mark.parametrize(
    ("question", "document_id", "required_text"),
    [
        (
            "Who owns and approves policy ETH-COC-01?",
            "ETH-COC-01",
            ("Arjun Malhotra", "Ananya Rao"),
        ),
        (
            "Who owns policy HR-PM-08, and when did the version take effect?",
            "HR-PM-08",
            ("Ritika Nair", "Kavita Menon", "2026-04-01"),
        ),
    ],
)
def test_explicit_document_control_tables_are_selected(
    question: str,
    document_id: str,
    required_text: tuple[str, ...],
) -> None:
    candidates = _candidates(document_id)
    baseline = ContextBuilder().build(candidates)
    assert all(term not in baseline["context"] for term in required_text)

    result, context = _compose(question, candidates)
    assert all(term in context for term in required_text)
    assert result["evidence_count"] == 8
    assert result["evidence_tokens"] <= 4000
    assert (
        result["citations"][0]["chunk_id"]
        == result["evaluation_trace"]["selected_evidence"][0]["chunk_id"]
    )
    assert result["evaluation_trace"]["retrieved_chunk_ids"] == [
        candidate["chunk"]["chunk_id"] for candidate in candidates
    ]


@pytest.mark.parametrize(
    "question",
    [
        "How do staff submit forms under ETH-COC-01?",
        "Who owns this policy?",
    ],
)
def test_no_priority_without_control_intent_and_explicit_document(
    question: str,
) -> None:
    candidates = _candidates("ETH-COC-01")
    result, _context = _compose(question, candidates)
    ranks = [
        evidence["rrf_rank"]
        for evidence in result["evaluation_trace"]["selected_evidence"]
    ]
    assert ranks == list(range(1, 9))
