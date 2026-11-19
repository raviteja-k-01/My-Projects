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
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from statistics import mean
from typing import Any

from google.genai import errors as genai_errors

from src.evaluation.loader import DEFAULT_GOLD_PATH, load_gold_questions
from src.evaluation.retrieval_metrics import DEFAULT_K_VALUES, evaluate_ranked_chunks
from src.retrieval.hybrid_retriever import HybridRetriever
from scripts.evaluate_hybrid_retrieval import get_chunk_ids, get_gold_ranks

DEFAULT_OUTPUT = Path("data/processed/hybrid_retrieval_live.jsonl")


def _build_result(gold: dict[str, Any], retrieved_chunk_ids: list[str],
                  *, provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    if len(retrieved_chunk_ids) != len(set(retrieved_chunk_ids)):
        raise ValueError(f"Duplicate retrieved chunk IDs for {gold['question_id']}")
    expected = gold["expected_chunk_ids"]
    result = {
        "question_id": gold["question_id"],
        "question": gold["question"],
        "question_type": gold["question_type"],
        "expected_chunk_ids": expected,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "gold_ranks": get_gold_ranks(retrieved_chunk_ids, expected),
        "metrics": evaluate_ranked_chunks(retrieved_chunk_ids, expected),
    }
    if provenance is not None:
        if (not isinstance(provenance, dict)
                or provenance.get("schema_version") != 1
                or provenance.get("retrieval_mode") != "hybrid"
                or provenance.get("retrieval_performed") is not True
                or not isinstance(provenance.get("corpus_sha256"), str)
                or len(provenance["corpus_sha256"]) != 64):
            raise ValueError("Invalid retrieval provenance")
        result["provenance"] = provenance
    return result


def retrieval_provenance(chunks_path: Path) -> dict[str, Any]:
    """Describe only a new retrieval; never backfill claims about old records."""
    from src.indexing.embedding_client import EMBEDDING_MODEL, EMBEDDING_DIMENSION
    from src.indexing.qdrant_store import COLLECTION_NAME
    from src.retrieval.rrf import RRF_K

    sources = [
        Path("src/retrieval/hybrid_retriever.py"), Path("src/retrieval/rrf.py"),
        Path("src/indexing/embedding_client.py"), Path("src/indexing/bm25_index.py"),
        Path("src/indexing/qdrant_store.py"),
    ]
    return {
        "schema_version": 1,
        "retrieval_mode": "hybrid",
        "retrieval_performed": True,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_sha256": sha256(chunks_path.read_bytes()).hexdigest(),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "query_normalization": "strip",
        "query_template": "task: question answering | query: {question}",
        "dense_collection": COLLECTION_NAME,
        "configured_vector_distance": "Cosine",
        "limits": {"dense": 20, "sparse": 20, "fused": 20},
        "fusion": {"algorithm": "RRF", "k": RRF_K},
        "source_sha256": {str(p): sha256(p.read_bytes()).hexdigest() for p in sources},
        "bm25_sha256": {
            p.name: sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path("data/bm25").iterdir()) if p.is_file()
        },
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
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Checkpoint line {line_number} is invalid JSON") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Checkpoint line {line_number} is not an object")
        question_id = record.get("question_id")
        if question_id not in by_id or question_id in completed:
            raise ValueError(f"Unexpected/duplicate checkpoint question: {question_id}")
        retrieved_ids = record.get("retrieved_chunk_ids")
        if not isinstance(retrieved_ids, list) or any(
            not isinstance(item, str) or not item.strip() for item in retrieved_ids
        ):
            raise ValueError(f"Invalid retrieved_chunk_ids for {question_id}")
        if record != _build_result(by_id[question_id], retrieved_ids,
                                   provenance=record.get("provenance")):
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


def run(*, output: Path = DEFAULT_OUTPUT, max_new: int = 1,
        gold_path: Path = DEFAULT_GOLD_PATH,
        chunks_path: Path = Path("data/processed/chunks.jsonl"),
        question_ids: set[str] | None = None) -> None:
    if type(max_new) is not int or max_new < 1:
        raise ValueError("--max-new must be positive")
    from src.evaluation.replay import load_replay_inputs

    gold_records = load_gold_questions(gold_path)
    if question_ids is not None and question_ids - {g["question_id"] for g in gold_records}:
        raise ValueError("Unknown requested question ID")
    completed, chunks = load_replay_inputs(
        gold=gold_records, checkpoint_path=output, chunks_path=chunks_path, allow_empty=True,
    )
    print(f"GOLD QUESTIONS: {len(gold_records)}; CHECKPOINTED: {len(completed)}")

    retriever: HybridRetriever | None = None
    new_queries = 0
    try:
        for index, gold in enumerate(gold_records, start=1):
            question_id = gold["question_id"]
            if question_ids is not None and question_id not in question_ids:
                continue
            if question_id in completed:
                print(f"[{index}/{len(gold_records)}] {question_id}: REUSED CHECKPOINT")
                continue
            if new_queries >= max_new:
                break
            provenance = retrieval_provenance(chunks_path)
            if retriever is None:
                retriever = HybridRetriever()
            print(f"[{index}/{len(gold_records)}] {question_id}: QUERYING", flush=True)
            # Gold answerability and expected chunks never control retrieval.
            results = retriever.retrieve(
                query=gold["question"], dense_limit=20, sparse_limit=20, fused_limit=20
            )
            retrieved_ids = get_chunk_ids(results)
            for item in results:
                if item["chunk_id"] not in chunks or item.get("chunk") != chunks[item["chunk_id"]]:
                    raise ValueError("Retrieved payload differs from verified corpus")
            new_queries += 1
            completed[question_id] = _build_result(gold, retrieved_ids, provenance=provenance)
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
        print("PROCESSING: COMPLETE; no questions have retrieval gold, so ranking metrics are unavailable.")
        return
    print("\nHYBRID RETRIEVAL SUMMARY")
    print(f"Evaluable questions: {len(evaluable)}")
    print(f"Questions without retrieval gold: {len(completed) - len(evaluable)}")
    print(f"MRR: {mean(record['metrics']['mrr'] for record in evaluable):.4f}")
    for k in DEFAULT_K_VALUES:
        print(f"Hit@{k}: {mean(r['metrics']['hit_at_k'][f'hit@{k}'] for r in evaluable):.4f}")
        print(f"Recall@{k}: {mean(r['metrics']['recall_at_k'][f'recall@{k}'] for r in evaluable):.4f}")
    print("HYBRID RETRIEVAL PROCESSING: COMPLETE (no quality threshold asserted)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-new", type=int, default=1,
                        help="Maximum new Gemini query embeddings this run (default: 1)")
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    args = parser.parse_args()
    try:
        run(output=args.output, max_new=args.max_new, chunks_path=args.chunks)
    except genai_errors.ClientError as exc:
        if getattr(exc, "code", None) != 429:
            raise
        print(
            "STOPPED: Gemini embedding quota exhausted. Existing evaluation "
            "checkpoints remain saved. Resume only after quota is available.",
            flush=True,
        )
        raise SystemExit(75) from None
    except genai_errors.ServerError as exc:
        print(f"STOPPED: embedding service HTTP {exc.code}; checkpoints preserved.")
        raise SystemExit(75) from None


if __name__ == "__main__":
    main()
