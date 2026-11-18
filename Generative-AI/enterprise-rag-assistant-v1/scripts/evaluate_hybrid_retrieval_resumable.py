"""Evaluate hybrid retrieval against gold questions with safe checkpoints.

Run from the project root:
    python -m scripts.evaluate_hybrid_retrieval_resumable

Results are saved after each question to an ignored local path by default.
A quota failure preserves completed records. Rerun the same command after the
quota resets to evaluate only the missing questions. Does not edit Qdrant.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

from google.genai import errors as genai_errors

from src.evaluation.loader import load_gold_questions
from src.evaluation.retrieval_metrics import DEFAULT_K_VALUES, evaluate_ranked_chunks
from src.retrieval.hybrid_retriever import HybridRetriever
from scripts.evaluate_hybrid_retrieval import get_chunk_ids, get_gold_ranks

DEFAULT_OUTPUT = Path("data/processed/hybrid_retrieval_live.jsonl")


def _build_result(gold: dict[str, Any], retrieved_chunk_ids: list[str]) -> dict[str, Any]:
    expected = gold["expected_chunk_ids"]
    return {
        "question_id": gold["question_id"],
        "question": gold["question"],
        "question_type": gold["question_type"],
        "expected_chunk_ids": expected,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "gold_ranks": get_gold_ranks(retrieved_chunk_ids, expected),
        "metrics": evaluate_ranked_chunks(retrieved_chunk_ids, expected),
    }


def load_checkpoint(output: Path, gold_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Refuse stale/incompatible output instead of silently reusing it."""
    if not output.exists():
        return {}

    by_id = {record["question_id"]: record for record in gold_records}
    completed: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(output.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"Checkpoint line {line_number} is not an object")
        question_id = record.get("question_id")
        if question_id not in by_id or question_id in completed:
            raise ValueError(f"Unexpected/duplicate checkpoint question: {question_id}")
        retrieved_ids = record.get("retrieved_chunk_ids")
        if not isinstance(retrieved_ids, list) or any(not isinstance(item, str) for item in retrieved_ids):
            raise ValueError(f"Invalid retrieved_chunk_ids for {question_id}")
        if record != _build_result(by_id[question_id], retrieved_ids):
            raise ValueError(f"Stale/inconsistent checkpoint for {question_id}; do not reuse it")
        completed[question_id] = record
    return completed


def save_checkpoint(output: Path, completed: dict[str, dict[str, Any]], gold_records: list[dict[str, Any]]) -> None:
    """Atomically replace a small JSONL file after each successful question."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for gold in gold_records:
            record = completed.get(gold["question_id"])
            if record is not None:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output)


def run(*, output: Path = DEFAULT_OUTPUT, max_new: int | None = None) -> None:
    if max_new is not None and max_new < 1:
        raise ValueError("--max-new must be positive")
    gold_records = load_gold_questions()
    completed = load_checkpoint(output, gold_records)
    print(f"GOLD QUESTIONS: {len(gold_records)}; CHECKPOINTED: {len(completed)}")

    retriever: HybridRetriever | None = None
    new_queries = 0
    try:
        for index, gold in enumerate(gold_records, start=1):
            question_id = gold["question_id"]
            if question_id in completed:
                print(f"[{index}/{len(gold_records)}] {question_id}: REUSED CHECKPOINT")
                continue
            if not gold["expected_chunk_ids"]:
                retrieved_ids: list[str] = []
                print(f"[{index}/{len(gold_records)}] {question_id}: NOT RETRIEVAL-EVALUABLE")
            else:
                if max_new is not None and new_queries >= max_new:
                    break
                if retriever is None:
                    retriever = HybridRetriever()
                print(f"[{index}/{len(gold_records)}] {question_id}: QUERYING", flush=True)
                results = retriever.retrieve(
                    query=gold["question"], dense_limit=20, sparse_limit=20, fused_limit=20
                )
                retrieved_ids = get_chunk_ids(results)
                new_queries += 1
            completed[question_id] = _build_result(gold, retrieved_ids)
            save_checkpoint(output, completed, gold_records)
            print(f"  SAVED: {len(completed)}/{len(gold_records)}", flush=True)
    finally:
        if retriever is not None:
            retriever.close()

    print(f"CHECKPOINT FILE: {output}")
    if len(completed) != len(gold_records):
        print(f"INCOMPLETE: {len(completed)}/{len(gold_records)} saved; rerun to resume.")
        return

    evaluable = [record for record in completed.values() if record["metrics"]["evaluable"]]
    if not evaluable:
        raise ValueError("No evaluable questions")
    print("\nHYBRID RETRIEVAL SUMMARY")
    print(f"Evaluable questions: {len(evaluable)}")
    print(f"Skipped questions: {len(completed) - len(evaluable)}")
    print(f"MRR: {mean(record['metrics']['mrr'] for record in evaluable):.4f}")
    for k in DEFAULT_K_VALUES:
        print(f"Hit@{k}: {mean(r['metrics']['hit_at_k'][f'hit@{k}'] for r in evaluable):.4f}")
        print(f"Recall@{k}: {mean(r['metrics']['recall_at_k'][f'recall@{k}'] for r in evaluable):.4f}")
    print("HYBRID RETRIEVAL EVALUATION: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-new", type=int, default=None,
                        help="Optional limit on new Gemini query embeddings this run")
    args = parser.parse_args()
    try:
        run(output=args.output, max_new=args.max_new)
    except genai_errors.ClientError as exc:
        if getattr(exc, "code", None) != 429:
            raise
        print(
            "STOPPED: Gemini embedding quota exhausted. Existing evaluation "
            "checkpoints remain saved. Resume only after quota is available.",
            flush=True,
        )
        raise SystemExit(75) from None


if __name__ == "__main__":
    main()
