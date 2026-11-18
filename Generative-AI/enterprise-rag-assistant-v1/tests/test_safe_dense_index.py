"""Offline tests for safe dense indexing: no Gemini or Qdrant credentials."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from scripts import build_dense_index as build
from src.indexing.qdrant_store import QdrantStore


@dataclass
class FakeChunk:
    chunk_id: str
    text: str
    token_count: int = 200


def _payloads(count: int = 5) -> dict[int, dict]:
    return build.expected_payloads([
        FakeChunk(f"TEST-{i}", f"Text {i}", 200)
        for i in range(1, count + 1)
    ])


def test_resume_uses_verified_ids_not_count():
    original = _payloads(5)
    observed = {2: original[2], 5: original[5]}
    assert build.plan_missing_ids(original, observed) == [1, 3, 4]


def test_equal_count_does_not_hide_mismatched_text():
    original = _payloads(3)
    existing = {k: dict(v) for k, v in original.items()}
    existing[2]["text"] = "OLD CORPUS"
    with pytest.raises(ValueError, match="mismatched payload IDs"):
        build.plan_missing_ids(original, existing)


def test_unexpected_existing_point_refused():
    expected = _payloads(2)
    with pytest.raises(ValueError, match="Unexpected IDs"):
        build.plan_missing_ids(
            expected,
            {1: expected[1], 2: expected[2], 3: {"text": "stale"}},
        )


def test_batcher_respects_request_count_and_estimated_tokens():
    expected = _payloads(7)
    batches = build.make_batches(
        list(expected),
        expected,
        batch_size=4,
        estimated_token_budget=450,
    )
    assert batches == [[1, 2], [3, 4], [5, 6], [7]]
    assert sorted(i for batch in batches for i in batch) == list(expected)


def test_batcher_rejects_invalid_token_estimates():
    expected = _payloads(1)
    expected[1]["token_count"] = 0
    with pytest.raises(ValueError, match="Invalid token"):
        build.make_batches([1], expected)


def test_batcher_rejects_oversized_input():
    expected = _payloads(1)
    expected[1]["token_count"] = 3000
    with pytest.raises(ValueError, match="exceeds estimated"):
        build.make_batches([1], expected)


class FakeStore:
    def __init__(self, existing):
        self.data = dict(existing)
        self.uploads = []
        self.ensure_calls = 0
        self.reads = 0

    def ensure_collection(self):
        self.ensure_calls += 1

    def existing_point_payloads(self):
        self.reads += 1
        return dict(self.data)

    def upsert_points(self, point_ids, vectors, payloads):
        self.uploads.append(list(point_ids))
        for point_id, payload in zip(point_ids, payloads):
            self.data[point_id] = payload


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return [[0.25] * 768 for _ in texts]


def test_run_index_fills_holes_without_touching_existing(monkeypatch):
    chunks = [
        FakeChunk("A", "one"),
        FakeChunk("B", "two"),
        FakeChunk("C", "three"),
        FakeChunk("D", "four"),
    ]
    expected = build.expected_payloads(chunks)
    store = FakeStore({2: expected[2], 4: expected[4]})
    embedder = FakeEmbedder()
    monkeypatch.setattr(build.time, "sleep", lambda _: None)

    build.run_index(chunks, embedder, store)

    assert store.uploads == [[1, 3]]
    assert embedder.calls == [["one", "three"]]
    assert len(store.data) == 4
    assert store.reads == 2


def test_run_index_rejects_stale_payload_before_embedding():
    chunks = [FakeChunk("A", "new")]
    store = FakeStore({1: {"chunk_id": "A", "text": "old", "token_count": 200}})
    embedder = FakeEmbedder()

    with pytest.raises(ValueError, match="mismatched payload IDs"):
        build.run_index(chunks, embedder, store)

    assert store.uploads == []
    assert embedder.calls == []


def test_run_index_noop_when_everything_already_exists():
    chunks = [FakeChunk("A", "one"), FakeChunk("B", "two")]
    store = FakeStore(build.expected_payloads(chunks))
    embedder = FakeEmbedder()

    build.run_index(chunks, embedder, store)

    assert not embedder.calls
    assert not store.uploads


class FakeQdrantClient:
    def __init__(self, records):
        self.records = records
        self.calls = []

    def scroll(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["offset"] is None:
            return self.records[:1], 1 if len(self.records) > 1 else None
        return self.records[1:], None


def _point(point_id, *, vector=None, payload=None):
    return SimpleNamespace(
        id=point_id,
        vector=[0.01] * 768 if vector is None else vector,
        payload={"chunk_id": str(point_id)} if payload is None else payload,
    )


def _store_with_points(points):
    store = object.__new__(QdrantStore)
    store.collection_name = "enterprise_rag_chunks"
    store.client = FakeQdrantClient(points)
    return store


def test_qdrant_audit_reads_all_pages_and_checks_vectors():
    store = _store_with_points([_point(1), _point(2)])
    assert store.existing_point_payloads() == {
        1: {"chunk_id": "1"},
        2: {"chunk_id": "2"},
    }
    assert len(store.client.calls) == 2
    assert all(c["with_vectors"] is True for c in store.client.calls)
    assert all(c["with_payload"] is True for c in store.client.calls)


def test_qdrant_audit_rejects_wrong_vector_size():
    store = _store_with_points([_point(1, vector=[0.1])])
    with pytest.raises(ValueError, match="Unexpected vector dimension"):
        store.existing_point_payloads()


def test_qdrant_audit_rejects_unexpected_id_type():
    store = _store_with_points([_point("uuid-id")])
    with pytest.raises(ValueError, match="Unexpected Qdrant point ID"):
        store.existing_point_payloads()


def test_embedding_client_uses_separate_content_objects_and_768_dims():
    """Regression: Embedding 2 aggregates bare parts, not Content objects."""
    from google.genai import types
    from src.indexing.embedding_client import GeminiEmbeddingClient

    class FakeModels:
        def __init__(self):
            self.calls = []

        def embed_content(self, *, model, contents, config):
            self.calls.append((model, contents, config))
            return SimpleNamespace(embeddings=[
                SimpleNamespace(values=[0.1] * 768)
                for _ in contents
            ])

    client = object.__new__(GeminiEmbeddingClient)
    client.client = SimpleNamespace(models=FakeModels())
    vectors = client.embed_documents(["first", "second"])

    assert len(vectors) == 2
    assert all(len(v) == 768 for v in vectors)
    model, contents, config = client.client.models.calls[0]
    assert model == "gemini-embedding-2"
    assert all(isinstance(c, types.Content) for c in contents)
    assert [c.parts[0].text for c in contents] == [
        "title: none | text: first",
        "title: none | text: second",
    ]
    assert config.output_dimensionality == 768


def test_embedding_client_refuses_aggregated_or_missing_vector():
    from src.indexing.embedding_client import GeminiEmbeddingClient

    class FakeModels:
        def embed_content(self, **kwargs):
            return SimpleNamespace(embeddings=[
                SimpleNamespace(values=[0.1] * 768)
            ])

    client = object.__new__(GeminiEmbeddingClient)
    client.client = SimpleNamespace(models=FakeModels())
    with pytest.raises(ValueError, match="Embedding count"):
        client.embed_documents(["first", "second"])
