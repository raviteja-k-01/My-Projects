"""Reconstruct exact ranked chunk payloads from a verified hybrid checkpoint.

This replays previously retrieved rankings. It does not claim that a new
embedding query was run or that a historic ranking represents the current
model version. Corpus SHA-256 is verified against the already indexed build.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from scripts.build_dense_index import verify_corpus
from scripts.evaluate_hybrid_retrieval_resumable import load_checkpoint


def load_replay_inputs(
    *,
    gold: list[dict[str, Any]],
    checkpoint_path: Path,
    chunks_path: Path,
    allow_empty: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Strictly validate the complete corpus and existing recorded rankings."""
    corpus = verify_corpus(chunks_path)
    by_chunk = {chunk.chunk_id: asdict(chunk) for chunk in corpus}
    if len(by_chunk) != len(corpus):
        raise ValueError("Duplicate chunk IDs in verified corpus")
    checkpoints = load_checkpoint(checkpoint_path, gold)
    if not checkpoints and not allow_empty:
        raise ValueError("No saved hybrid retrieval checkpoints available")
    for qid, record in checkpoints.items():
        provenance = record.get("provenance")
        if provenance is not None:
            from scripts.build_dense_index import EXPECTED_CHUNKS_SHA256
            if provenance["corpus_sha256"] != EXPECTED_CHUNKS_SHA256:
                raise ValueError(f"Checkpoint corpus SHA-256 mismatch for {qid}")
        ids = record["retrieved_chunk_ids"]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate retrieved chunk IDs for {qid}")
        if any(chunk_id not in by_chunk for chunk_id in ids):
            raise ValueError(f"Unknown retrieved chunk ID in checkpoint for {qid}")
    return checkpoints, by_chunk


def is_replayable(record: dict[str, Any]) -> bool:
    """Old no-gold/empty records were skips, not evidence of empty retrieval."""
    return bool(record["retrieved_chunk_ids"] or record.get("expected_chunk_ids")
                or record.get("provenance", {}).get("retrieval_performed") is True)


def replay_candidates(
    question_id: str,
    checkpoints: dict[str, dict[str, Any]],
    by_chunk: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if question_id not in checkpoints:
        raise ValueError(f"No hybrid retrieval checkpoint for {question_id}")
    ids = checkpoints[question_id]["retrieved_chunk_ids"]
    if not is_replayable(checkpoints[question_id]):
        raise ValueError(f"Legacy skipped retrieval cannot establish empty evidence for {question_id}")
    return [
        {
            "chunk_id": chunk_id,
            "rank": rank,
            "chunk": by_chunk[chunk_id],
        }
        for rank, chunk_id in enumerate(ids, start=1)
    ]
