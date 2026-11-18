from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import qdrant_backup
from src.indexing.qdrant_store import QdrantStore


class FakeClient:
    def __init__(self, rows=None):
        self.rows = {} if rows is None else dict(rows)

    def scroll(self, *, collection_name, limit, offset, with_payload, with_vectors):
        assert collection_name == "enterprise_rag_chunks"
        assert with_payload and with_vectors
        ids = sorted(i for i in self.rows if offset is None or i > offset)
        page = ids[:limit]
        records = [
            SimpleNamespace(
                id=i,
                payload=self.rows[i]["payload"],
                vector=self.rows[i]["vector"],
            )
            for i in page
        ]
        return records, (page[-1] if len(ids) > limit else None)


class FakeStore:
    collection_name = "enterprise_rag_chunks"

    def __init__(self, rows=None):
        self.client = FakeClient(rows)

    def _validate_vector(self, vector):
        return QdrantStore._validate_vector(self, vector)

    def ensure_collection(self):
        return None

    def upsert_points(self, *, point_ids, vectors, payloads):
        for point_id, vector, payload in zip(point_ids, vectors, payloads):
            self.client.rows[point_id] = {
                "id": point_id,
                "vector": vector,
                "payload": payload,
            }


def _source_rows():
    expected = {
        1: {"chunk_id": "A", "text": "first chunk"},
        2: {"chunk_id": "B", "text": "second chunk"},
    }
    rows = {
        i: {"id": i, "payload": payload, "vector": [0.001 * i] * 768}
        for i, payload in expected.items()
    }
    return expected, rows


def test_backup_export_validate_and_resumable_restore(tmp_path):
    expected, rows = _source_rows()
    source = tmp_path / "chunks.jsonl"
    source.write_text('corpus', encoding="utf-8")
    backup = tmp_path / "secure" / "vectors.jsonl"
    qdrant_backup.export_backup(FakeStore(rows), expected, source, backup)
    loaded = qdrant_backup.load_backup(backup, source, expected)
    assert loaded == rows
    partially_restored = FakeStore({1: rows[1]})
    qdrant_backup.restore_backup(partially_restored, loaded, expected)
    assert partially_restored.client.rows == rows


def test_backup_rejects_tampered_bytes(tmp_path):
    expected, rows = _source_rows()
    source = tmp_path / "chunks.jsonl"
    source.write_text("corpus")
    backup = tmp_path / "vectors.jsonl"
    qdrant_backup.export_backup(FakeStore(rows), expected, source, backup)
    with backup.open("a") as handle:
        handle.write("tampered\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        qdrant_backup.load_backup(backup, source, expected)


def test_backup_rejects_incompatible_existing_vectors(tmp_path):
    expected, rows = _source_rows()
    different = {
        1: {"id": 1, "payload": rows[1]["payload"], "vector": [0.9] * 768}
    }
    with pytest.raises(ValueError, match="Incompatible existing Qdrant vector"):
        qdrant_backup.restore_backup(FakeStore(different), rows, expected)


def test_backup_rejects_mismatched_payload():
    expected, rows = _source_rows()
    rows[1]["payload"] = {"chunk_id": "BAD"}
    with pytest.raises(ValueError, match="payload mismatch"):
        qdrant_backup.validate_backup_rows(rows, expected)
