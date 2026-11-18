"""Offline contract test: degraded retrieval must be disclosed, not disguised."""
from __future__ import annotations

from src.rag.rag_pipeline import EnterpriseRAGPipeline
from src.retrieval.context_builder import ContextBuilder
from src.generation.citation_builder import CitationBuilder


class FakeRetriever:
    def retrieve_with_mode(self, *, query, allow_sparse_fallback, **kwargs):
        assert allow_sparse_fallback is True
        assert query == "What is the policy?"
        return [
            {
                "rank": 1,
                "chunk_id": "POLICY-B001",
                "chunk": {
                    "chunk_id": "POLICY-B001",
                    "document_id": "POLICY",
                    "document_name": "Synthetic policy",
                    "text": "Incidents must be reported immediately.",
                    "page_start": 1,
                    "page_end": 1,
                    "token_count": 12,
                },
            }
        ], "bm25_fallback"


class FakeGenerator:
    def generate(self, question, context):
        assert "[Evidence 1]" in context
        return {
            "answer": "Report incidents immediately. [Evidence 1]",
            "model": "fake-offline",
            "usage": {},
        }


def test_pipeline_discloses_bm25_fallback_and_real_citation():
    pipeline = EnterpriseRAGPipeline.__new__(EnterpriseRAGPipeline)
    pipeline.retriever = FakeRetriever()
    pipeline.context_builder = ContextBuilder()
    pipeline.citation_builder = CitationBuilder()
    pipeline.generator = FakeGenerator()

    response = pipeline.answer(" What is the policy? ")
    assert response["retrieval_mode"] == "bm25_fallback"
    assert response["answer"] == "Report incidents immediately. [Evidence 1]"
    assert response["citation_count"] == 1
    assert response["citations"][0]["chunk_id"] == "POLICY-B001"
    assert response["evidence_count"] == 1
