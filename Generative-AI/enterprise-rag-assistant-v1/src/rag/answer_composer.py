"""Shared answer assembly for live retrieval and verified retrieval replay.

The caller supplies ranked retrieval results. This module does not access
Qdrant, invoke Gemini embeddings, or change retrieval rankings.
Only the injected generator makes a generation-model request.
"""
from __future__ import annotations

import time
from typing import Any

from src.generation.citation_builder import CitationBuilder
from src.retrieval.context_builder import ContextBuilder


INSUFFICIENT_EVIDENCE_ANSWER = (
    "There is insufficient evidence in the available policy documents to answer this question."
)


def compose_answer(
    *,
    question: str,
    candidates: list[dict[str, Any]],
    retrieval_mode: str,
    context_builder: ContextBuilder,
    generator: Any,
    citation_builder: CitationBuilder,
    retrieval_seconds: float | None,
    retrieval_source: str,
    evaluation_trace: bool = False,
    total_start: float | None = None,
) -> dict[str, Any]:
    """Use the exact same context/generation/citation path in both modes.

    For saved retrieval replay, retrieval_seconds is None, because the
    historical retrieval timing is unavailable. total_seconds then measures
    only the context + generation + citation work done during this invocation.
    """
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty")
    if retrieval_mode not in ("hybrid", "bm25_fallback"):
        raise ValueError("Unrecognized retrieval mode")
    if retrieval_source not in ("live", "saved_hybrid_checkpoint"):
        raise ValueError("Unrecognized retrieval source")
    if retrieval_source == "saved_hybrid_checkpoint" and retrieval_mode != "hybrid":
        raise ValueError("Saved hybrid checkpoint may not be relabeled as BM25")

    begin = time.perf_counter() if total_start is None else total_start
    # The builder enforces the eight-evidence cap. Let it refill skipped duplicate
    # or oversized chunks from the rest of the retrieved ranking.
    context_result = context_builder.build(candidates=candidates)
    if context_result["context"]:
        generation_start = time.perf_counter()
        generated = generator.generate(question=question, context=context_result["context"])
        generation_seconds = time.perf_counter() - generation_start
        citation_result = citation_builder.build(
            answer=generated["answer"], evidence=context_result["evidence"],
        )
    else:
        # This decision uses only the selected evidence, never evaluation labels.
        generated = {"model": None, "usage": {}}
        generation_seconds = 0.0
        citation_result = {
            "answer": INSUFFICIENT_EVIDENCE_ANSWER, "citations": [], "citation_count": 0,
        }

    response: dict[str, Any] = {
        "question": question,
        "retrieval_mode": retrieval_mode,
        "retrieval_source": retrieval_source,
        "answer": citation_result["answer"],
        "citations": citation_result["citations"],
        "citation_count": citation_result["citation_count"],
        "evidence_count": context_result["evidence_count"],
        "evidence_tokens": context_result["total_tokens"],
        "model": generated["model"],
        "usage": generated["usage"],
        "latency": {
            "retrieval_seconds": retrieval_seconds,
            "reranking_seconds": 0.0,
            "generation_seconds": generation_seconds,
            "total_seconds": time.perf_counter() - begin,
        },
    }
    if evaluation_trace:
        response["evaluation_trace"] = {
            "retrieved_chunk_ids": [item["chunk_id"] for item in candidates],
            "selected_evidence": context_result["evidence"],
        }
    return response
