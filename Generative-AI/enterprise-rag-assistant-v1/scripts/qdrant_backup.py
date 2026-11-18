"""Verified vector backup and safe restore for this 1,985-point corpus.

Export is read-only. Restore defaults to offline validation; --apply is
required to create/update Qdrant points. No Gemini API calls in either mode.

    python -m scripts.qdrant_backup export --chunks PATH_TO_CHUNKS
    python -m scripts.qdrant_backup restore --chunks PATH_TO_CHUNKS
    python -m scripts.qdrant_backup restore --chunks PATH_TO_CHUNKS --apply
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

from src.indexing.embedding_client import EMBEDDING_DIMENSION, EMBEDDING_MODEL
from src.indexing.qdrant_store import COLLECTION_NAME, QdrantStore
from scripts.build_dense_index import expected_payloads, plan_missing_ids, verify_corpus

DEFAULT_BACKUP = Path("data/qdrant_validation/enterprise_rag_chunks_backup.jsonl")


def _digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_path(backup: Path) -> Path:
    return backup.with_name(backup.name + ".manifest.json")


def read_points(store: QdrantStore) -> dict[int, dict[str, Any]]:
    """Read IDs, complete payloads, and all vectors without writing."""
    rows: dict[int, dict[str, Any]] = {}
    offset = None
    seen_offsets: set[str] = set()

    while True:
        points, next_offset = store.client.scroll(
            collection_name=store.collection_name,
            limit=64,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for point in points:
            if type(point.id) is not int or point.id < 1 or point.id in rows:
                raise ValueError("Invalid or duplicated Qdrant point ID")
            if not isinstance(point.payload, dict) or not isinstance(point.vector, list):
                raise ValueError(f"Missing Qdrant payload/vector at ID {point.id}")
            store._validate_vector(point.vector)
            rows[point.id] = {
                "id": point.id,
                "payload": point.payload,
                "vector": point.vector,
            }
        if next_offset is None:
            return rows
        token = str(next_offset)
        if token in seen_offsets or next_offset == offset:
            raise RuntimeError("Qdrant pagination did not advance")
        seen_offsets.add(token)
        offset = next_offset


def validate_backup_rows(rows: dict[int, dict[str, Any]], expected: dict[int, dict]) -> None:
    if set(rows) != set(expected):
        raise ValueError(
            f"Backup IDs differ from corpus: {len(rows)} points, "
            f"{len(set(expected) - set(rows))} missing, {len(set(rows) - set(expected))} extra"
        )
    for point_id in sorted(expected):
        row = rows[point_id]
        if row.get("id") != point_id or row.get("payload") != expected[point_id]:
            raise ValueError(f"Backup payload mismatch at point ID {point_id}")
        vector = row.get("vector")
        if not isinstance(vector, list) or len(vector) != EMBEDDING_DIMENSION:
            raise ValueError(f"Invalid backup vector dimensions at point ID {point_id}")
        from math import isfinite
        if not all(isinstance(x, (int, float)) and isfinite(x) for x in vector):
            raise ValueError(f"Invalid backup vector values at point ID {point_id}")


def export_backup(
    store: QdrantStore,
    expected: dict[int, dict],
    chunks_path: Path,
    backup: Path,
) -> None:
    rows = read_points(store)
    validate_backup_rows(rows, expected)

    backup.parent.mkdir(parents=True, exist_ok=True)
    temporary = backup.with_name(backup.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for point_id in sorted(rows):
            handle.write(json.dumps(rows[point_id], separators=(",", ":"), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(backup)

    manifest = {
        "backup_sha256": _digest(backup),
        "chunks_sha256": _digest(chunks_path),
        "collection": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "dimensions": EMBEDDING_DIMENSION,
        "points": len(rows),
    }
    manifest_file = _manifest_path(backup)
    tmp_manifest = manifest_file.with_name(manifest_file.name + ".tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_manifest.replace(manifest_file)
    print(f"BACKUP VERIFIED: {len(rows)} points, SHA-256 {manifest['backup_sha256']}")
    print(f"Backup: {backup}")
    print(f"Manifest: {manifest_file}")
    print("Keep both files private and outside Git.")


def load_backup(backup: Path, chunks_path: Path, expected: dict[int, dict]) -> dict[int, dict[str, Any]]:
    manifest_file = _manifest_path(backup)
    if not backup.is_file() or not manifest_file.is_file():
        raise FileNotFoundError("Both backup JSONL and its manifest are required")

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    expected_manifest = {
        "chunks_sha256": _digest(chunks_path),
        "collection": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "dimensions": EMBEDDING_DIMENSION,
        "points": len(expected),
    }
    for key, value in expected_manifest.items():
        if manifest.get(key) != value:
            raise ValueError(f"Backup manifest incompatible: {key}")

    if manifest.get("backup_sha256") != _digest(backup):
        raise ValueError("Backup SHA-256 mismatch; do not restore")

    rows: dict[int, dict[str, Any]] = {}
    with backup.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            point_id = record.get("id")
            if type(point_id) is not int or point_id in rows:
                raise ValueError("Duplicate or invalid backup point ID")
            rows[point_id] = record
    validate_backup_rows(rows, expected)
    return rows


def restore_backup(store: QdrantStore, rows: dict[int, dict[str, Any]], expected: dict[int, dict]) -> None:
    store.ensure_collection()
    observed = read_points(store)
    # Do not overwrite *any* point with a different payload OR vector.
    for point_id, row in observed.items():
        if point_id not in rows or row["payload"] != rows[point_id]["payload"]:
            raise ValueError(f"Incompatible existing Qdrant point ID {point_id}")
        stored = row["vector"]
        backup_vector = rows[point_id]["vector"]
        if len(stored) != len(backup_vector) or any(
            abs(a - b) > 1e-5 for a, b in zip(stored, backup_vector)
        ):
            raise ValueError(f"Incompatible existing Qdrant vector ID {point_id}")

    plan_missing_ids(expected, {i: record["payload"] for i, record in observed.items()})
    missing = sorted(set(expected) - set(observed))
    print(f"RESTORE: {len(observed)} compatible existing, {len(missing)} missing.")
    for start in range(0, len(missing), 32):
        batch_ids = missing[start : start + 32]
        store.upsert_points(
            point_ids=batch_ids,
            payloads=[rows[i]["payload"] for i in batch_ids],
            vectors=[rows[i]["vector"] for i in batch_ids],
        )
        print(f"RESTORE CHECKPOINT: {min(start + 32, len(missing))}/{len(missing)}")

    final = read_points(store)
    if set(final) != set(rows):
        raise RuntimeError("Qdrant restore incomplete")
    for point_id, record in rows.items():
        if final[point_id]["payload"] != record["payload"] or any(
            abs(a - b) > 1e-5 for a, b in zip(final[point_id]["vector"], record["vector"])
        ):
            raise RuntimeError(f"Qdrant post-restore mismatch at {point_id}")
    print(f"RESTORE PASSED: {len(final)} exact IDs/payloads and matching vectors.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode in ("export", "restore"):
        operation = modes.add_parser(mode)
        operation.add_argument("--chunks", type=Path, required=True)
        operation.add_argument("--backup", type=Path, default=DEFAULT_BACKUP)
        if mode == "restore":
            operation.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    chunks = verify_corpus(args.chunks)
    expected = expected_payloads(chunks)
    if args.mode == "restore":
        rows = load_backup(args.backup, args.chunks, expected)
        print(f"BACKUP VERIFIED OFFLINE: {len(rows)} points; integrity + corpus match.")
        if not args.apply:
            print("RESTORE DRY RUN: no Qdrant access, Gemini calls, or writes.")
            return

    store = QdrantStore()
    try:
        if args.mode == "export":
            export_backup(store, expected, args.chunks, args.backup)
        else:
            restore_backup(store, rows, expected)
    finally:
        store.close()


if __name__ == "__main__":
    main()
