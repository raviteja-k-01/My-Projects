from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.api.rate_limit import SlidingWindowRateLimiter
from src.indexing import embedding_client
from src.retrieval import hybrid_retriever


class FakeEmbeddingModels:
    def __init__(self):
        self.calls = []

    def embed_content(self, **kwargs):
        self.calls.append(kwargs)
        contents = kwargs["contents"]
        count = len(contents) if isinstance(contents, list) else 1
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0.05] * 768) for _ in range(count)]
        )


def test_query_cache_and_defensive_vector_copy():
    models = FakeEmbeddingModels()
    client = embedding_client.GeminiEmbeddingClient(
        client=SimpleNamespace(models=models)
    )
    a = client.embed_query("  What is SEV-1?  ")
    a[0] = 99.0
    b = client.embed_query("What is SEV-1?")
    assert b[0] == 0.05
    assert len(b) == 768
    assert len(models.calls) == 1
    assert models.calls[0]["model"] == "gemini-embedding-2"
    assert models.calls[0]["config"].output_dimensionality == 768
    client.embed_query("Another question")
    assert len(models.calls) == 2


def test_documents_separate_embeddings_not_query_cached():
    models = FakeEmbeddingModels()
    client = embedding_client.GeminiEmbeddingClient(
        client=SimpleNamespace(models=models)
    )
    assert len(client.embed_documents(["first", "second"])) == 2
    assert len(models.calls) == 1
    assert isinstance(models.calls[0]["contents"], list)
    assert len(models.calls[0]["contents"]) == 2


def test_explicit_primary_or_secondary_key(monkeypatch):
    called_keys = []
    monkeypatch.setenv("GEMINI_API_KEY", "primary")
    monkeypatch.setenv("GEMINI_API_KEY_1", "secondary")
    monkeypatch.setattr(
        embedding_client.genai, "Client",
        lambda *, api_key: called_keys.append(api_key) or SimpleNamespace(),
    )
    embedding_client.GeminiEmbeddingClient(key_env="GEMINI_API_KEY")
    embedding_client.GeminiEmbeddingClient(key_env="GEMINI_API_KEY_1")
    assert called_keys == ["primary", "secondary"]


def test_invalid_key_selector_rejected():
    with pytest.raises(ValueError, match="Invalid GEMINI_EMBEDDING_KEY_ENV"):
        embedding_client.GeminiEmbeddingClient(key_env="UNEXPECTED_SECRET")


def _retriever_with_fake_results(monkeypatch, error=None):
    retriever = hybrid_retriever.HybridRetriever.__new__(
        hybrid_retriever.HybridRetriever
    )

    def dense(*, query, limit):
        if error:
            raise error
        return [
            {"chunk_id": "D", "rank": 1, "score": 1.0, "chunk": {"chunk_id": "D"}}
        ]

    def sparse(*, query, limit):
        return [
            {"chunk_id": "S", "rank": 1, "score": 1.0, "chunk": {"chunk_id": "S"}}
        ]

    monkeypatch.setattr(retriever, "_dense_search", dense)
    monkeypatch.setattr(retriever, "_sparse_search", sparse)
    return retriever


def test_strict_evaluation_never_downgrades_to_sparse(monkeypatch):
    class FakeClientError(Exception):
        code = 429

    monkeypatch.setattr(hybrid_retriever.genai_errors, "ClientError", FakeClientError)
    retriever = _retriever_with_fake_results(monkeypatch, FakeClientError())
    with pytest.raises(FakeClientError):
        retriever.retrieve("Test question")


def test_application_fallback_is_explicitly_labeled(monkeypatch):
    class FakeClientError(Exception):
        code = 429

    monkeypatch.setattr(hybrid_retriever.genai_errors, "ClientError", FakeClientError)
    retriever = _retriever_with_fake_results(monkeypatch, FakeClientError())
    results, mode = retriever.retrieve_with_mode(
        "Test question", allow_sparse_fallback=True
    )
    assert mode == "bm25_fallback"
    assert [result["chunk_id"] for result in results] == ["S"]


def test_unauthorized_credentials_do_not_trigger_fallback(monkeypatch):
    class FakeClientError(Exception):
        code = 403

    monkeypatch.setattr(hybrid_retriever.genai_errors, "ClientError", FakeClientError)
    retriever = _retriever_with_fake_results(monkeypatch, FakeClientError())
    with pytest.raises(FakeClientError):
        retriever.retrieve_with_mode("Test question", allow_sparse_fallback=True)


def test_sliding_window_limits_and_recovers():
    limiter = SlidingWindowRateLimiter(limit=2, window_seconds=60)
    assert limiter.allow("ip1", now=0)
    assert limiter.allow("ip1", now=1)
    assert not limiter.allow("ip1", now=2)
    assert limiter.allow("ip2", now=2)
    assert limiter.allow("ip1", now=60)


def test_api_enforces_rate_limit_without_using_gemini(monkeypatch):
    import src.api.app as module

    class FakePipeline:
        def answer(self, question):
            return {"answer": "ok", "citations": []}

        def close(self):
            pass

    monkeypatch.setattr(module, "EnterpriseRAGPipeline", FakePipeline)
    monkeypatch.setattr(
        module, "rate_limiter", SlidingWindowRateLimiter(limit=1, window_seconds=60)
    )
    with TestClient(module.app) as client:
        assert client.post("/ask", json={"question": "First"}).status_code == 200
        second = client.post("/ask", json={"question": "Second"})
        assert second.status_code == 429
        assert second.headers["Retry-After"] == "60"
        assert client.post("/ask", json={"question": "   "}).status_code == 422
        assert client.post("/ask", json={"question": "x" * 1001}).status_code == 422
