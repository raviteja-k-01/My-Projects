"""Rebuild the verified 1,985-point dense index, safely and resumably.

Start in the project root. Requires the reproducible chunks.jsonl produced
by CI, QDRANT_URL / QDRANT_API_KEY, and GEMINI_API_KEY to write vectors.

    python -m scripts.build_dense_index --chunks path/to/chunks.jsonl
    python -m scripts.build_dense_index --chunks path/to/chunks.jsonl --apply

Without --apply, only local corpus integrity is checked: no API calls.
With --apply, existing Qdrant point IDs, full payloads AND vector lengths
are audited before reusing any points. No deletions or collection resets.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import time

from google.genai import errors

from src.core.models import ChunkRecord
from src.indexing.chunk_loader import load_indexing_chunks
from src.indexing.embedding_client import GeminiEmbeddingClient
from src.indexing.qdrant_store import QdrantStore


# From the CI-verified offline build on the reproducible-retrieval-artifacts PR.
EXPECTED_CHUNKS_SHA256 = (
    "41b7f8ff1d5db169a7cf5c945dbcc5d655578b6e25bea665c10dc7888a83abf4"
)
EXPECTED_CHUNKS = 1985

# AI Studio free tier shown for this project: 100 RPM, 30k TPM, 1000 RPD.
# The chunk tokenizer uses characters/4 as an ESTIMATE, not Google's count.
# Conservative token-budget pacing reduces risk; API still enforces actual TPM.
MAX_BATCH_SIZE = 16
MAX_ESTIMATED_TOKENS_PER_BATCH = 2500
BATCH_DELAY_SECONDS = 12.5
RATE_LIMIT_WAIT_SECONDS = 70
MAX_429_RETRIES = 3


def verify_corpus(chunks_path: Path) -> list[ChunkRecord]:
    """Refuse partial or changed inputs before touching the vector store."""
    if not chunks_path.is_file():
        raise FileNotFoundError(f"Missing chunks file: {chunks_path}")
    digest = sha256(chunks_path.read_bytes()).hexdigest()
    if digest != EXPECTED_CHUNKS_SHA256:
        raise ValueError(
            "Chunks SHA-256 differs from the CI-verified build. "
            f"Expected {EXPECTED_CHUNKS_SHA256}, got {digest}. "
            "Do not mix retrieval artifacts from different builds."
        )
    chunks = load_indexing_chunks(chunks_path)
    if len(chunks) != EXPECTED_CHUNKS:
        raise ValueError(
            f"Wrong corpus size: expected {EXPECTED_CHUNKS}, got {len(chunks)}"
        )
    return chunks


def expected_payloads(chunks: list[ChunkRecord]) -> dict[int, dict]:
    """Stable historical point ID contract: 1-based position in chunk file."""
    return {
        index: asdict(chunk)
        for index, chunk in enumerate(chunks, start=1)
    }


def plan_missing_ids(
    expected: dict[int, dict],
    observed: dict[int, dict],
) -> list[int]:
    """Check every existing payload; never infer completeness from count.

    A point ID may be present with outdated text even if the final count
    looks correct. Refuse that database rather than silently mixing builds.
    """
    unexpected = sorted(observed.keys() - expected.keys())
    mismatched = sorted(
        point_id
        for point_id in observed.keys() & expected.keys()
        if observed[point_id] != expected[point_id]
    )
    if unexpected or mismatched:
        raise ValueError(
            "Qdrant contains incompatible points; no indexing performed. "
            f"Unexpected IDs (first 5): {unexpected[:5]}; "
            f"mismatched payload IDs (first 5): {mismatched[:5]}"
        )
    return sorted(expected.keys() - observed.keys())


def make_batches(
    missing_ids: list[int],
    expected: dict[int, dict],
    *,
    batch_size: int = MAX_BATCH_SIZE,
    estimated_token_budget: int = MAX_ESTIMATED_TOKENS_PER_BATCH,
) -> list[list[int]]:
    """Bound both request batch size and approximate token volume."""
    if batch_size < 1 or estimated_token_budget < 1:
        raise ValueError("Batch size and token budget must be positive")
    batches: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0

    for point_id in missing_ids:
        token_estimate = expected[point_id]["token_count"]
        if not isinstance(token_estimate, int) or token_estimate <= 0:
            raise ValueError(f"Invalid token estimate for point {point_id}")
        # The request adds a small "title: none | text:" prefix.
        request_estimate = token_estimate + 12
        if request_estimate > estimated_token_budget:
            raise ValueError(
                f"Point {point_id} exceeds estimated per-batch token budget"
            )
        if current and (
            len(current) >= batch_size
            or current_tokens + request_estimate > estimated_token_budget
        ):
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(point_id)
        current_tokens += request_estimate

    if current:
        batches.append(current)
    return batches


def _daily_limit(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in (
        "perday", "per_day", "per day", "requests per day",
        "daily quota", "rpd",
    ))


def embed_with_retry(
    embedder: GeminiEmbeddingClient,
    texts: list[str],
) -> list[list[float]]:
    """Retry bounded per-minute 429s; stop safely for daily or unknown errors."""
    for attempt in range(MAX_429_RETRIES + 1):
        try:
            return embedder.embed_documents(texts)
        except errors.ClientError as exc:
            if exc.code != 429:
                raise
            if _daily_limit(exc):
                raise RuntimeError(
                    "Daily Gemini embedding quota exhausted; prior Qdrant "
                    "batches remain saved. Resume after quota reset."
                ) from exc
            if attempt >= MAX_429_RETRIES:
                raise RuntimeError(
                    "Gemini 429 persisted after bounded retries. Prior "
                    "Qdrant batches remain saved; inspect quota before resuming."
                ) from exc
            print(
                f"Gemini 429: waiting {RATE_LIMIT_WAIT_SECONDS} seconds "
                f"(retry {attempt + 1}/{MAX_429_RETRIES})"
            )
            time.sleep(RATE_LIMIT_WAIT_SECONDS)

    raise AssertionError("Unreachable retry state")


def run_index(
    chunks: list[ChunkRecord],
    embedder: GeminiEmbeddingClient,
    store: QdrantStore,
    *,
    max_batches: int | None = None,
) -> None:
    """Validate before writing, resume by individual verified point ID."""
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive")
    expected = expected_payloads(chunks)
    store.ensure_collection()
    existing = store.existing_point_payloads()
    missing = plan_missing_ids(expected, existing)
    batches = make_batches(missing, expected)

    print(
        f"Verified corpus: {len(expected)}; already indexed: {len(existing)}; "
        f"missing: {len(missing)}; planned embedding requests: {len(batches)}"
    )
    if not missing:
        print("PASSED: All 1,985 point IDs, payloads, and vectors verified.")
        return

    work_batches = batches if max_batches is None else batches[:max_batches]
    for batch_number, batch_ids in enumerate(work_batches, start=1):
        payloads = [expected[i] for i in batch_ids]
        vectors = embed_with_retry(
            embedder,
            [payload["text"] for payload in payloads],
        )
        if len(vectors) != len(batch_ids):
            raise ValueError(
                "Gemini returned an unexpected embedding count; "
                "batch was NOT uploaded"
            )
        store.upsert_points(
            point_ids=batch_ids,
            vectors=vectors,
            payloads=payloads,
        )
        print(
            f"Verified progress: {len(existing) + sum(map(len, batches[:batch_number]))}"
            f"/{len(expected)}; batch {batch_number}/{len(work_batches)}"
        )
        if batch_number < len(work_batches):
            time.sleep(BATCH_DELAY_SECONDS)

    # Do not trust count alone; re-read IDs, full payloads, and vector lengths.
    complete = store.existing_point_payloads()
    remaining = plan_missing_ids(expected, complete)
    if remaining and max_batches is not None:
        print(
            f"CHECKPOINT VERIFIED: {len(complete)} points, "
            f"{len(remaining)} remain. Resume with --apply."
        )
        return
    if remaining:
        raise RuntimeError(f"Qdrant final audit missing {len(remaining)} points")
    if len(complete) != len(expected):
        raise RuntimeError("Qdrant final point count mismatch")
    print(f"PASSED: all {len(expected)} point IDs, payloads and vectors verified.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks", type=Path, default=Path("data/processed/chunks.jsonl"),
        help="Path to the exact CI-verified chunks.jsonl artifact",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Opt in to Gemini calls and writes to Qdrant (otherwise local-only)",
    )
    parser.add_argument(
        "--max-batches", type=int, default=None,
        help="Optional checkpoint: stop after N successfully uploaded batches",
    )
    args = parser.parse_args()

    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be positive")
    chunks = verify_corpus(args.chunks)
    expected = expected_payloads(chunks)
    batches = make_batches(sorted(expected), expected)
    print(f"CORPUS VERIFIED: {len(chunks)} chunks, SHA-256 match.")
    print(
        f"Full-rebuild plan: {len(batches)} embedding requests, "
        f"{MAX_BATCH_SIZE} max inputs/request, "
        f"{MAX_ESTIMATED_TOKENS_PER_BATCH} estimated tokens/request."
    )
    if not args.apply:
        print("DRY RUN: no Gemini calls, no Qdrant access or writes.")
        print("Pass --apply only after configuring a new Qdrant cluster and keys.")
        return

    embedder = GeminiEmbeddingClient()
    store = QdrantStore()
    try:
        run_index(chunks, embedder, store, max_batches=args.max_batches)
    finally:
        store.close()


if __name__ == "__main__":
    main()
