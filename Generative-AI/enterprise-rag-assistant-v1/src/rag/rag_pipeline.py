"""Enterprise RAG pipeline: retrieve, then assemble a grounded answer."""
from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from typing import Any

from src.generation.answer_generator import GroundedAnswerGenerator, SYSTEM_PROMPT
from src.generation.failover_generator import FailoverAnswerGenerator, GroqAnswerGenerator
from src.generation.citation_builder import CitationBuilder
from src.rag.answer_composer import compose_answer
from src.retrieval.context_builder import ContextBuilder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.rag.query_router import direct_response, lacks_relevant_evidence, quick_route

DEFAULT_DENSE_LIMIT = 20
DEFAULT_SPARSE_LIMIT = 20
DEFAULT_FUSED_LIMIT = 20


class EnterpriseRAGPipeline:
    """Live RAG: hybrid retrieval, selected evidence, selected LLM, citations.

    Replays of verified historical hybrid retrieval use the *same*
    compose_answer function without generating a new query embedding.
    """

    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self.context_builder = ContextBuilder(max_chunks=8, max_context_tokens=4000)
        # Live API generation is configurable. Replays keep their explicit
        # experiment generator; the indexed Gemini embeddings are unchanged.
        load_dotenv()
        provider = os.getenv("RAG_GENERATION_PROVIDER", "gemini").strip().lower()
        if provider not in {"gemini", "groq"}:
            raise ValueError("RAG_GENERATION_PROVIDER must be gemini or groq")
        if provider == "groq":
            # Groq-only: never initialize or invoke Gemini answer generation.
            self.generator = GroqAnswerGenerator(system_prompt=SYSTEM_PROMPT)
        else:
            # Legacy Gemini primary, with optional one-shot Groq failover.
            primary = GroundedAnswerGenerator()
            flag = os.getenv("GROQ_FALLBACK_ENABLED", "false").strip().lower()
            if flag not in {"", "0", "false", "no", "1", "true", "yes"}:
                raise ValueError("GROQ_FALLBACK_ENABLED must be true or false")
            self.generator = primary
            if flag in {"1", "true", "yes"}:
                self.generator = FailoverAnswerGenerator(
                    primary=primary,
                    fallback=GroqAnswerGenerator(system_prompt=SYSTEM_PROMPT),
                )
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
        # Fast, explicit chat/out-of-domain routes use no embeddings, Qdrant,
        # BM25, or generation API. Keep strict evaluation on the original path.
        if not evaluation_trace:
            route = quick_route(question)
            if route is not None:
                return direct_response(
                    question, response_type=route.response_type,
                    answer=route.answer, started=total_start,
                )

        retrieval_start = time.perf_counter()
        candidates, retrieval_mode = self.retriever.retrieve_with_mode(
            query=question,
            dense_limit=DEFAULT_DENSE_LIMIT,
            sparse_limit=DEFAULT_SPARSE_LIMIT,
            fused_limit=DEFAULT_FUSED_LIMIT,
            allow_sparse_fallback=allow_sparse_fallback,
            comparison_coverage=True,
        )
        retrieval_seconds = time.perf_counter() - retrieval_start

        # Conservative, local post-retrieval guard: when no selected evidence
        # shares even a substantive word with a non-policy question, do not
        # spend a generation request on likely irrelevant context. Do not use
        # this heuristic in benchmark traces or to reject policy-intent queries.
        if not evaluation_trace and candidates:
            selected = self.context_builder.build(candidates)["evidence"]
            if lacks_relevant_evidence(question, selected):
                from src.rag.query_router import NO_RELEVANT_EVIDENCE
                return direct_response(
                    question, response_type="insufficient_evidence",
                    answer=NO_RELEVANT_EVIDENCE, started=total_start,
                    retrieval_mode=retrieval_mode, retrieval_seconds=retrieval_seconds,
                )

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
