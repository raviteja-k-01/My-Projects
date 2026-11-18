"""Enterprise RAG pipeline: retrieve, then assemble a grounded answer."""
from __future__ import annotations

import time
from typing import Any

from src.generation.answer_generator import GroundedAnswerGenerator
from src.generation.citation_builder import CitationBuilder
from src.rag.answer_composer import compose_answer
from src.retrieval.context_builder import ContextBuilder
from src.retrieval.hybrid_retriever import HybridRetriever

DEFAULT_DENSE_LIMIT = 20
DEFAULT_SPARSE_LIMIT = 20
DEFAULT_FUSED_LIMIT = 20


class EnterpriseRAGPipeline:
    """Live RAG: hybrid retrieval, selected evidence, Gemini, citations.

    Replays of verified historical hybrid retrieval use the *same*
    compose_answer function without generating a new query embedding.
    """

    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self.context_builder = ContextBuilder(max_chunks=8, max_context_tokens=4000)
        self.generator = GroundedAnswerGenerator()
        self.citation_builder = CitationBuilder()

    def answer(
        self,
        question: str,
        *,
        evaluation_trace: bool = False,
        allow_sparse_fallback: bool = True,
    ) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")

        total_start = time.perf_counter()
        retrieval_start = time.perf_counter()
        candidates, retrieval_mode = self.retriever.retrieve_with_mode(
            query=question,
            dense_limit=DEFAULT_DENSE_LIMIT,
            sparse_limit=DEFAULT_SPARSE_LIMIT,
            fused_limit=DEFAULT_FUSED_LIMIT,
            allow_sparse_fallback=allow_sparse_fallback,
        )
        retrieval_seconds = time.perf_counter() - retrieval_start
        return compose_answer(
            question=question,
            candidates=candidates,
            retrieval_mode=retrieval_mode,
            retrieval_source="live",
            context_builder=self.context_builder,
            generator=self.generator,
            citation_builder=self.citation_builder,
            retrieval_seconds=retrieval_seconds,
            evaluation_trace=evaluation_trace,
            total_start=total_start,
        )

    def close(self) -> None:
        self.retriever.close()
