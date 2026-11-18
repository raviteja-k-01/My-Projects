"""Read-only audit: check every local chunk against the existing Qdrant payload.

Do not run a dense reindex just to repair a mismatch. Inspect the difference
between the regenerated corpus and the already-deployed dense index first.

Run from project root after rebuilding artifacts, with QDRANT_URL and
QDRANT_API_KEY provided in the environment or a local .env file:
    python -m scripts.verify_qdrant_alignment
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.indexing.chunk_loader import load_indexing_chunks
from src.indexing.qdrant_store import QdrantStore


def compare_payloads(
    expected: dict[str, dict[str, Any]],
    observed: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str], list[str]]:
    """Return missing, unexpected, and metadata/text-mismatched chunk IDs."""
    missing = sorted(expected.keys() - observed.keys())
    unexpected = sorted(observed.keys() - expected.keys())
    different = sorted(
        chunk_id
        for chunk_id in expected.keys() & observed.keys()
        if expected[chunk_id] != observed[chunk_id]
    )
    return missing, unexpected, different


def fetch_payloads(store: QdrantStore) -> dict[str, dict[str, Any]]:
    """Read all Qdrant payloads, without fetching vectors or modifying points."""
    chunks: dict[str, dict[str, Any]] = {}
    offset = None

    while True:
        points, next_offset = store.client.scroll(
            collection_name=store.collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload
            if not isinstance(payload, dict):
                raise ValueError(f"Qdrant point {point.id} has no payload")
            chunk_id = payload.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError(f"Qdrant point {point.id} lacks chunk_id")
            if chunk_id in chunks:
                raise ValueError(f"Duplicate chunk_id in Qdrant: {chunk_id}")
            chunks[chunk_id] = payload

        if next_offset is None:
            break
        if next_offset == offset:
            raise RuntimeError("Qdrant scroll pagination did not advance")
        offset = next_offset

    return chunks


def main() -> None:
    expected_chunks = load_indexing_chunks()
    expected = {chunk.chunk_id: asdict(chunk) for chunk in expected_chunks}
    if len(expected) != len(expected_chunks):
        raise ValueError("Duplicate local chunk IDs")

    store = QdrantStore()
    try:
        observed = fetch_payloads(store)
    finally:
        store.close()

    missing, unexpected, different = compare_payloads(expected, observed)
    print(f"Local chunks: {len(expected)}")
    print(f"Qdrant payloads: {len(observed)}")
    print(f"Missing from Qdrant: {len(missing)}")
    print(f"Unexpected in Qdrant: {len(unexpected)}")
    print(f"Different text or metadata: {len(different)}")

    if missing or unexpected or different:
        for label, ids in (
            ("Missing", missing),
            ("Unexpected", unexpected),
            ("Different", different),
        ):
            if ids:
                print(f"{label} sample chunk IDs: {ids[:10]}")
        raise SystemExit(
            "Qdrant/local chunk alignment FAILED. "
            "Do not serve this BM25 index with the existing Qdrant collection."
        )

    print("Qdrant/local chunk alignment PASSED (full payload equality).")


if __name__ == "__main__":
    main()
