import time
from typing import Any

from src.generation.answer_generator import GroundedAnswerGenerator
from src.generation.citation_builder import CitationBuilder
from src.retrieval.context_builder import ContextBuilder
from src.retrieval.hybrid_retriever import HybridRetriever

DEFAULT_DENSE_LIMIT = 20
DEFAULT_SPARSE_LIMIT = 20
DEFAULT_FUSED_LIMIT = 20
DEFAULT_RERANK_LIMIT = 8


class EnterpriseRAGPipeline:
    """
    Complete enterprise RAG pipeline.

    Query
    → Hybrid retrieval
    → RRF
    → Cross-encoder reranking
    → Context building
    → Gemini generation
    → Citation mapping
    """

    def __init__(self) -> None:
        self.retriever = HybridRetriever()

        self.context_builder = ContextBuilder(
            max_chunks=8,
            max_context_tokens=4000,
        )

        self.generator = GroundedAnswerGenerator()

        self.citation_builder = CitationBuilder()

    def answer(
        self,
        question: str,
    ) -> dict[str, Any]:

        question = question.strip()

        if not question:
            raise ValueError(
                "Question cannot be empty"
            )

        total_start = time.perf_counter()

        # -------------------------------------------------
        # 1. Hybrid retrieval:
        # Dense + BM25 + RRF
        # -------------------------------------------------

        retrieval_start = time.perf_counter()

        candidates = self.retriever.retrieve(
            query=question,
            dense_limit=DEFAULT_DENSE_LIMIT,
            sparse_limit=DEFAULT_SPARSE_LIMIT,
            fused_limit=DEFAULT_FUSED_LIMIT,
        )

        retrieval_seconds = (
            time.perf_counter()
            - retrieval_start
        )

        # -------------------------------------------------
        # 2. Select top hybrid candidates
        #
        # Cross-encoder reranking is disabled in V1 because
        # formal evaluation showed worse retrieval quality.
        # -------------------------------------------------

        selected_candidates = candidates[
            :DEFAULT_RERANK_LIMIT
        ]

        rerank_seconds = 0.0
        # -------------------------------------------------
        # 3. Build clean evidence context
        # -------------------------------------------------

        context_result = self.context_builder.build(
            candidates=selected_candidates,
        )

        if not context_result["context"]:
            raise ValueError(
                "No usable evidence was found"
            )

        # -------------------------------------------------
        # 4. Generate grounded answer with Gemini
        # -------------------------------------------------

        generation_start = time.perf_counter()

        generated = self.generator.generate(
            question=question,
            context=context_result["context"],
        )

        generation_seconds = (
            time.perf_counter()
            - generation_start
        )

        # -------------------------------------------------
        # 5. Convert [Evidence N] into real citations
        # -------------------------------------------------

        citation_result = self.citation_builder.build(
            answer=generated["answer"],
            evidence=context_result["evidence"],
        )

        total_seconds = (
            time.perf_counter()
            - total_start
        )

        # -------------------------------------------------
        # 6. Final structured response
        # -------------------------------------------------

        return {
            "question": question,

            "answer": citation_result["answer"],

            "citations": citation_result[
                "citations"
            ],

            "citation_count": citation_result[
                "citation_count"
            ],

            "evidence_count": context_result[
                "evidence_count"
            ],

            "evidence_tokens": context_result[
                "total_tokens"
            ],

            "model": generated["model"],

            "usage": generated["usage"],

            "latency": {
                "retrieval_seconds": (
                    retrieval_seconds
                ),
                "reranking_seconds": (
                    rerank_seconds
                ),
                "generation_seconds": (
                    generation_seconds
                ),
                "total_seconds": (
                    total_seconds
                ),
            },
        }

    def close(self) -> None:
        """
        Close external resources.
        """

        self.retriever.close()