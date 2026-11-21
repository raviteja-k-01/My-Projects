"""Offline regression tests for local routing and conservative evidence checks."""
from __future__ import annotations

import pytest

from src.rag.query_router import lacks_relevant_evidence, quick_route
from src.rag.rag_pipeline import EnterpriseRAGPipeline
from src.retrieval.context_builder import ContextBuilder
from src.generation.citation_builder import CitationBuilder


EXAMPLE_CHUNK = {
    "chunk_id": "POLICY-TEST-1",
    "document_id": "SEC-POL-01",
    "document_name": "Synthetic security policy",
    "text": "Privileged access must be approved by the accountable manager.",
    "page_start": 1,
    "page_end": 1,
    "token_count": 16,
}


class FakeRetriever:
    def __init__(self, *, forbid: bool = False):
        self.calls = 0
        self.forbid = forbid

    def retrieve_with_mode(self, **kwargs):
        self.calls += 1
        if self.forbid:
            pytest.fail("Router must not call retrieval or embeddings")
        return [{
            "chunk_id": EXAMPLE_CHUNK["chunk_id"],
            "rank": 1,
            "chunk": EXAMPLE_CHUNK,
        }], "hybrid"


class FakeGenerator:
    def __init__(self, *, forbid: bool = False):
        self.calls = 0
        self.forbid = forbid

    def generate(self, question, context):
        self.calls += 1
        if self.forbid:
            pytest.fail("Router must not invoke either LLM")
        assert "[Evidence 1]" in context
        return {
            "answer": "Privileged access requires approval. [Evidence 1]",
            "model": "test-offline",
            "usage": {},
        }


def make_pipeline(*, forbid_retrieval=False, forbid_generation=False):
    pipeline = EnterpriseRAGPipeline.__new__(EnterpriseRAGPipeline)
    pipeline.retriever = FakeRetriever(forbid=forbid_retrieval)
    pipeline.generator = FakeGenerator(forbid=forbid_generation)
    pipeline.context_builder = ContextBuilder()
    pipeline.citation_builder = CitationBuilder()
    return pipeline


@pytest.mark.parametrize(("question", "expected"), [
    ("hi how are you??", "greeting"),
    ("Hey there!", "greeting"),
    ("good morning", "greeting"),
    ("hey assistant did modi win 2024 elections?", "out_of_scope"),
    ("Who won the FIFA World Cup?", "out_of_scope"),
    ("What is the weather in Delhi today?", "out_of_scope"),
    ("Please tell me a joke.", "out_of_scope"),
    ("What is the capital of France?", "out_of_scope"),
    ("Write Python code for Fibonacci", "out_of_scope"),
])
def test_obvious_unrelated_requests_are_handled_without_network(question, expected):
    pipeline = make_pipeline(forbid_retrieval=True, forbid_generation=True)
    result = pipeline.answer(question)
    assert result["response_type"] == expected
    assert result["retrieval_mode"] == "not_run"
    assert result["retrieval_source"] == "not_applicable"
    assert result["model"] is None
    assert result["citations"] == []
    assert result["generation_fallback_used"] is False
    assert result["latency"]["generation_seconds"] == 0.0
    assert pipeline.retriever.calls == pipeline.generator.calls == 0


@pytest.mark.parametrize("question", [
    "Hi, how do I revoke privileged access?",
    "What is our policy on election-related social media?",
    "What controls are required when granting privileged production access?",
    "What should happen if an administrator leaves a project?",
    "Can I work from home?",
    "How do I report a lost laptop?",
    "What is the SEV-1 escalation policy?",
])
def test_policy_intent_never_fast_rejected(question):
    assert quick_route(question) is None
    assert not lacks_relevant_evidence(
        question, [{"text": "Unrelated content about governance."}]
    )


def test_policy_question_preserves_retrieval_generation_and_citations():
    pipeline = make_pipeline()
    result = pipeline.answer("What is the privileged access policy?")
    assert result.get("response_type", "grounded_answer") == "grounded_answer"
    assert result["retrieval_mode"] == "hybrid"
    assert result["citation_count"] == 1
    assert result["citations"][0]["chunk_id"] == EXAMPLE_CHUNK["chunk_id"]
    assert result["model"] == "test-offline"
    assert pipeline.retriever.calls == pipeline.generator.calls == 1


def test_no_matching_evidence_abstains_without_generation():
    pipeline = make_pipeline(forbid_generation=True)
    result = pipeline.answer("Why is Jupiter red?")
    assert result["response_type"] == "insufficient_evidence"
    assert result["retrieval_mode"] == "hybrid"
    assert "couldn't find relevant evidence" in result["answer"]
    assert result["citations"] == []
    assert result["model"] is None
    assert result["latency"]["generation_seconds"] == 0.0
    assert pipeline.retriever.calls == 1
    assert pipeline.generator.calls == 0


def test_evidence_word_overlap_is_not_mistaken_for_verified_factual_support():
    # This function is only a cheap *negative* filter, not an accuracy metric.
    assert lacks_relevant_evidence(
        "Why is Jupiter red?", [{"text": "Privileged manager access."}]
    )
    assert not lacks_relevant_evidence(
        "Is Jupiter red?", [{"text": "Jupiter appears in this document."}]
    )


def test_strict_evaluation_disables_live_only_routing_and_guard():
    pipeline = make_pipeline()
    result = pipeline.answer("hi", evaluation_trace=True, allow_sparse_fallback=False)
    assert result["retrieval_mode"] == "hybrid"
    assert result["model"] == "test-offline"
    assert result["evaluation_trace"]["selected_evidence"][0]["chunk_id"] == EXAMPLE_CHUNK["chunk_id"]
    assert pipeline.retriever.calls == pipeline.generator.calls == 1
