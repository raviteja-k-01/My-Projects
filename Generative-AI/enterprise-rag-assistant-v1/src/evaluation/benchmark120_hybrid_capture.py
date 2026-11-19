"""Bounded, resumable *strict hybrid* retrieval for source-verified benchmarks.

No query embeddings are requested unless explicitly authorized. Never uses
BM25 fallback and never derives relevance labels from matched source text.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

from src.evaluation.benchmark120_evidence import read_jsonl, sha256


def _question_digest(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def _chunk_ids(corpus_path: Path) -> set[str]:
    ids: set[str] = set()
    for chunk in read_jsonl(corpus_path):
        cid = chunk.get("chunk_id")
        if not isinstance(cid, str) or not cid.strip() or cid in ids:
            raise ValueError(f"Missing or duplicate corpus chunk ID: {cid!r}")
        ids.add(cid)
    if not ids:
        raise ValueError("Corpus has no chunks")
    return ids


def _verified_checkpoint(
    checkpoint_path: Path,
    by_id: dict[str, dict[str, Any]],
    split: str,
    dataset_digest: str,
    corpus_digest: str,
    corpus_ids: set[str],
    limits: dict[str, int],
) -> dict[str, dict[str, Any]]:
    saved: dict[str, dict[str, Any]] = {}
    if not checkpoint_path.exists():
        return saved
    for row in read_jsonl(checkpoint_path):
        qid = row.get("question_id")
        if qid not in by_id or qid in saved:
            raise ValueError(f"Unknown/duplicate checkpoint question: {qid}")
        q = by_id[qid]
        ids = row.get("retrieved_chunk_ids")
        if (row.get("split") != split
            or row.get("question_sha256") != _question_digest(q["question"])
            or row.get("source_dataset_sha256") != dataset_digest
            or row.get("source_corpus_sha256") != corpus_digest
            or row.get("retrieval_mode") != "strict_hybrid"
            or row.get("retrieval_limits") != limits
            or not isinstance(ids, list)
            or any(not isinstance(cid, str) or cid not in corpus_ids for cid in ids)
            or len(ids) != len(set(ids))):
            raise ValueError(f"Stale or invalid strict-hybrid checkpoint: {qid}")
        saved[qid] = row
    return saved


def _save_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".benchmark120-hybrid-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def capture_hybrid(
    *,
    dataset_path: Path,
    corpus_path: Path,
    checkpoint_path: Path,
    questions: list[dict[str, Any]],
    split: str = "dev",
    max_new: int = 0,
    confirm_live: bool = False,
    allow_test: bool = False,
    dense_limit: int = 20,
    sparse_limit: int = 20,
    fused_limit: int = 20,
    retriever_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Preview with max_new=0; explicit opt-in is required for live calls.

    The strict retriever may consume one Gemini embedding per new query. Failures
    do not write a row; prior rows remain, with dataset/corpus/config hashes.
    """
    if split not in ("dev", "test"):
        raise ValueError("split must be 'dev' or 'test'")
    if type(max_new) is not int or max_new < 0:
        raise ValueError("max_new must be a nonnegative integer")
    limits = {"dense_limit": dense_limit, "sparse_limit": sparse_limit, "fused_limit": fused_limit}
    if any(type(v) is not int or v < 1 for v in limits.values()):
        raise ValueError("Retrieval limits must be positive integers")
    if max_new and not confirm_live:
        raise ValueError("Live embedding calls require --confirm-live; default is preview only")
    if split == "test" and max_new and not allow_test:
        raise ValueError("Test split is held out; live capture requires --unlock-test")

    subset = [q for q in questions if q["split"] == split]
    by_id = {q["question_id"]: q for q in subset}
    if not subset or len(subset) != len(by_id):
        raise ValueError("No questions or repeated question IDs in selected split")
    dataset_digest = sha256(dataset_path)
    corpus_digest = sha256(corpus_path)
    ids = _chunk_ids(corpus_path)
    saved = _verified_checkpoint(checkpoint_path, by_id, split, dataset_digest, corpus_digest, ids, limits)
    pending = [q for q in subset if q["question_id"] not in saved]
    report = {
        "split": split, "questions": len(subset), "checkpointed": len(saved),
        "pending": len(pending), "budget_new_query_embeddings": max_new,
        "checkpoint": str(checkpoint_path), "live_calls_enabled": bool(max_new and confirm_live),
        "mode": "strict_hybrid_no_sparse_fallback", "dataset_sha256": dataset_digest,
        "corpus_sha256": corpus_digest, "newly_saved": 0,
    }
    if not max_new or not pending:
        return report

    if retriever_factory is None:
        # Imported only after explicit live opt-in; preview cannot initialize providers.
        from src.retrieval.hybrid_retriever import HybridRetriever
        retriever_factory = HybridRetriever
    retriever = retriever_factory()
    try:
        for q in pending[:max_new]:
            try:
                # .retrieve is explicitly strict (no BM25-only service fallback).
                results = retriever.retrieve(query=q["question"], **limits)
            except Exception as exc:
                code = getattr(exc, "code", None)
                description = f"{type(exc).__name__} HTTP {code}" if code is not None else type(exc).__name__
                raise RuntimeError(
                    f"Strict hybrid retrieval stopped ({description}); prior checkpoint entries preserved"
                ) from exc
            ranked = []
            for item in results:
                if not isinstance(item, dict):
                    raise ValueError("Retriever returned a non-dictionary candidate")
                cid = item.get("chunk_id")
                if not isinstance(cid, str) or cid not in ids or cid in ranked:
                    raise ValueError(f"Invalid, unknown or duplicate retrieved chunk ID: {cid!r}")
                ranked.append(cid)
            row = {
                "question_id": q["question_id"], "split": split,
                "question_sha256": _question_digest(q["question"]),
                "source_dataset_sha256": dataset_digest,
                "source_corpus_sha256": corpus_digest,
                "retrieval_mode": "strict_hybrid", "retrieval_limits": limits,
                "retrieved_chunk_ids": ranked,
            }
            saved[q["question_id"]] = row
            _save_atomic(checkpoint_path, (saved[x["question_id"]] for x in subset if x["question_id"] in saved))
            report["newly_saved"] += 1
    finally:
        retriever.close()
    report["checkpointed"] = len(saved)
    report["pending"] = len(subset) - len(saved)
    return report
