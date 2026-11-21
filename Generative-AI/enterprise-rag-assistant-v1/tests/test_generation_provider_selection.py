"""Offline tests for generation-provider selection. No network/model calls."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.generation.failover_generator import GroqAnswerGenerator, FailoverAnswerGenerator
from src.rag import rag_pipeline
from scripts.deployment_preflight import PreflightError, validate_config


class NoNetworkRetriever:
    def __init__(self):
        self.calls = 0

    def retrieve_with_mode(self, **_kwargs):
        self.calls += 1
        return [], "hybrid"

    def close(self):
        pass


def test_groq_only_never_initializes_gemini_generation(monkeypatch):
    retriever = NoNetworkRetriever()
    monkeypatch.setenv("RAG_GENERATION_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "fake-local-test")
    monkeypatch.setenv("GROQ_FALLBACK_ENABLED", "true")  # Ignored in Groq-only mode.
    monkeypatch.setattr(rag_pipeline, "HybridRetriever", lambda: retriever)
    monkeypatch.setattr(
        rag_pipeline, "GroundedAnswerGenerator",
        lambda: pytest.fail("Groq-only must not create a Gemini generation client"),
    )
    pipeline = rag_pipeline.EnterpriseRAGPipeline()
    try:
        assert isinstance(pipeline.generator, GroqAnswerGenerator)
        assert pipeline.generator.model_name == "openai/gpt-oss-120b"
        answer = pipeline.answer("What is the security policy?")
        assert answer["model"] is None  # Empty evidence: no generation.
        assert retriever.calls == 1
    finally:
        pipeline.close()


def test_existing_gemini_fallback_mode_remains_available(monkeypatch):
    monkeypatch.setenv("RAG_GENERATION_PROVIDER", "gemini")
    monkeypatch.setenv("GROQ_FALLBACK_ENABLED", "true")
    monkeypatch.setenv("GROQ_API_KEY", "fake-local-test")
    monkeypatch.setattr(rag_pipeline, "HybridRetriever", NoNetworkRetriever)
    primary = SimpleNamespace(generate=lambda **_kw: None)
    monkeypatch.setattr(rag_pipeline, "GroundedAnswerGenerator", lambda: primary)
    pipeline = rag_pipeline.EnterpriseRAGPipeline()
    try:
        assert isinstance(pipeline.generator, FailoverAnswerGenerator)
        assert pipeline.generator.primary is primary
        assert isinstance(pipeline.generator.fallback, GroqAnswerGenerator)
    finally:
        pipeline.close()


def test_invalid_generation_provider_fails_closed(monkeypatch):
    monkeypatch.setenv("RAG_GENERATION_PROVIDER", "not-a-provider")
    monkeypatch.setattr(rag_pipeline, "HybridRetriever", NoNetworkRetriever)
    with pytest.raises(ValueError, match="RAG_GENERATION_PROVIDER"):
        rag_pipeline.EnterpriseRAGPipeline()


def test_groq_only_preflight_requires_groq_key():
    env = {
        "RAG_GENERATION_PROVIDER": "groq",
        "GEMINI_EMBEDDING_KEY_ENV": "GEMINI_API_KEY_1",
        "GEMINI_API_KEY_1": "embedding-only",
        "QDRANT_URL": "https://example.invalid",
        "QDRANT_API_KEY": "fake",
    }
    with pytest.raises(PreflightError, match="GROQ_API_KEY"):
        validate_config(env)
    validate_config({**env, "GROQ_API_KEY": "fake-groq"})


def test_groq_only_does_not_require_gemini_generation_key():
    validate_config({
        "RAG_GENERATION_PROVIDER": "groq",
        "GEMINI_EMBEDDING_KEY_ENV": "GEMINI_API_KEY_1",
        "GEMINI_API_KEY_1": "embedding-only",
        "GEMINI_GENERATION_KEY_ENV": "GEMINI_API_KEY",
        "GROQ_API_KEY": "fake-groq",
        "QDRANT_URL": "https://example.invalid",
        "QDRANT_API_KEY": "fake",
    })


def test_preflight_rejects_unknown_generation_provider():
    with pytest.raises(PreflightError, match="RAG_GENERATION_PROVIDER"):
        validate_config({
            "RAG_GENERATION_PROVIDER": "unknown",
            "GEMINI_API_KEY": "embedding-only",
            "QDRANT_URL": "https://example.invalid",
            "QDRANT_API_KEY": "fake",
        })
