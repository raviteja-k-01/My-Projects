"""Read-only comparison of regenerated chunks and existing Qdrant payloads.

Requires only Python's standard library. No Gemini calls, no Qdrant writes.

1. Download the `rag-v1-retrieval-artifacts` ZIP from the successful
   GitHub Actions run, and extract its `processed/chunks.jsonl`.
2. From project root, set environment variables QDRANT_URL and QDRANT_API_KEY
   securely (do not type them into public logs).
3. Run:
   python -m scripts.verify_qdrant_alignment --chunks path/to/chunks.jsonl

This proves payload equivalence, NOT equality of embedding vectors.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


COLLECTION_NAME = "enterprise_rag_chunks"


def compare_payloads(
    expected: dict[str, dict[str, Any]],
    observed: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str], list[str]]:
    """Find missing, unexpected and mismatched chunk IDs."""
    missing = sorted(expected.keys() - observed.keys())
    unexpected = sorted(observed.keys() - expected.keys())
    different = sorted(
        chunk_id
        for chunk_id in expected.keys() & observed.keys()
        if expected[chunk_id] != observed[chunk_id]
    )
    return missing, unexpected, different


def load_local_payloads(chunks_path: Path) -> dict[str, dict[str, Any]]:
    if not chunks_path.is_file():
        raise FileNotFoundError(f"Missing chunks file: {chunks_path}")

    chunks: dict[str, dict[str, Any]] = {}
    with chunks_path.open(encoding="utf-8") as stream:
        for line_num, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid chunk object on line {line_num}")
            chunk_id = payload.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError(f"Missing chunk ID on line {line_num}")
            if chunk_id in chunks:
                raise ValueError(f"Duplicate chunk ID in local data: {chunk_id}")
            chunks[chunk_id] = payload

    if not chunks:
        raise ValueError("Local corpus is empty")
    return chunks


def _qdrant_scroll_request(
    base_url: str,
    api_key: str,
    offset: str | int | None,
) -> dict[str, Any]:
    """Call the read-only scroll endpoint and return its result."""
    parts = urlsplit(base_url)
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
        raise ValueError("QDRANT_URL must be a valid HTTPS endpoint without credentials")
    if parts.query or parts.fragment:
        raise ValueError("QDRANT_URL must not contain query parameters or fragments")

    url = base_url.rstrip("/") + f"/collections/{COLLECTION_NAME}/points/scroll"
    body: dict[str, Any] = {
        "limit": 256,
        "with_payload": True,
        "with_vector": False,
    }
    if offset is not None:
        body["offset"] = offset
    request = Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "api-key": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"Qdrant read-only request failed: HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError("Qdrant request failed: check HTTPS URL and network") from exc

    if payload.get("status") != "ok" or not isinstance(payload.get("result"), dict):
        raise RuntimeError("Unexpected Qdrant scroll response")
    return payload["result"]


def fetch_qdrant_payloads(base_url: str, api_key: str) -> dict[str, dict[str, Any]]:
    chunks: dict[str, dict[str, Any]] = {}
    offset: str | int | None = None
    seen_offsets: set[str] = set()

    while True:
        result = _qdrant_scroll_request(base_url, api_key, offset)
        points = result.get("points")
        if not isinstance(points, list):
            raise RuntimeError("Qdrant response has no points list")

        for point in points:
            payload = point.get("payload")
            if not isinstance(payload, dict):
                raise ValueError("Qdrant point has no payload")
            chunk_id = payload.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError("Qdrant point lacks chunk_id")
            if chunk_id in chunks:
                raise ValueError(f"Duplicate chunk ID in Qdrant: {chunk_id}")
            chunks[chunk_id] = payload

        next_offset = result.get("next_page_offset")
        if next_offset is None:
            return chunks
        token = str(next_offset)
        if token in seen_offsets or next_offset == offset:
            raise RuntimeError("Qdrant pagination did not advance")
        seen_offsets.add(token)
        offset = next_offset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks", type=Path, default=Path("data/processed/chunks.jsonl"),
        help="Path to regenerated chunks.jsonl",
    )
    args = parser.parse_args()

    expected = load_local_payloads(args.chunks)
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url or not api_key:
        raise SystemExit("Missing QDRANT_URL or QDRANT_API_KEY environment variable")

    observed = fetch_qdrant_payloads(url, api_key)
    missing, unexpected, different = compare_payloads(expected, observed)

    print(f"Local chunks: {len(expected)}")
    print(f"Qdrant payloads: {len(observed)}")
    print(f"Missing from Qdrant: {len(missing)}")
    print(f"Unexpected in Qdrant: {len(unexpected)}")
    print(f"Changed text or metadata: {len(different)}")

    if missing or unexpected or different:
        for label, ids in (
            ("Missing", missing), ("Unexpected", unexpected), ("Changed", different),
        ):
            if ids:
                print(f"{label} sample IDs: {ids[:10]}")
        raise SystemExit("FAILED: Do not mix the new BM25 index with existing Qdrant data")

    print("PASSED: All local chunk payloads match Qdrant exactly.")
    print("Note: Stored embedding vectors were NOT independently verified.")


if __name__ == "__main__":
    main()
