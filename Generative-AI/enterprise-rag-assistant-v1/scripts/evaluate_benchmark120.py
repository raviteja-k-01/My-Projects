"""Offline, separate 120-question benchmark workflow.

This CLI never calls a model, embedding service, or network endpoint.
The 30-question gold evaluation remains unchanged. Source anchors are not
human-verified relevance labels; a label must be explicitly reviewed.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

from src.evaluation.benchmark120_evidence import audit, read_jsonl, sha256

DEFAULT_DATASET = Path("data/evaluation/rag_eval_120.jsonl")
ALLOWED_SPLITS = frozenset({"dev", "test"})


def load_dataset(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("Benchmark dataset is empty")
    seen: set[str] = set()
    for row in rows:
        qid = row.get("question_id")
        if not isinstance(qid, str) or not qid.strip() or qid in seen:
            raise ValueError(f"Empty or repeated question ID: {qid!r}")
        seen.add(qid)
        if row.get("split") not in ALLOWED_SPLITS:
            raise ValueError(f"{qid}: split must be dev or test")
        if type(row.get("answerable")) is not bool:
            raise ValueError(f"{qid}: answerable must be boolean")
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError(f"{qid}: missing question text")
        if not isinstance(row.get("expected_facts"), list) or not row["expected_facts"] or not all(
            isinstance(f, str) and f.strip() for f in row["expected_facts"]
        ):
            raise ValueError(f"{qid}: expected_facts must be nonempty strings")
        for field in ("expected_documents", "evidence_documents"):
            if not isinstance(row.get(field), list) or not all(
                isinstance(d, str) and d.strip() for d in row[field]
            ):
                raise ValueError(f"{qid}: {field} must be document ID strings")
    return rows


def summary(path: Path) -> dict[str, Any]:
    rows = load_dataset(path)
    return {
        "dataset_sha256": sha256(path),
        "question_count": len(rows),
        "by_split": dict(sorted(collections.Counter(q["split"] for q in rows).items())),
        "answerable_count": sum(q["answerable"] for q in rows),
        "unanswerable_count": sum(not q["answerable"] for q in rows),
        "expected_fact_prompts": sum(len(q["expected_facts"]) for q in rows),
        "note": "Question counts and source matches do not constitute validated gold labels.",
    }


def verified_audit(dataset: Path, audit_dir: Path) -> dict[str, Any]:
    report = json.loads((audit_dir / "summary.json").read_text(encoding="utf-8"))
    if report.get("dataset_sha256") != sha256(dataset):
        raise ValueError("Audit dataset hash mismatch: rerun audit on this exact dataset")
    return report


def _write_jsonl_new(path: Path, records: list[dict[str, Any]]) -> None:
    # Exclusive create: do not silently overwrite human work.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as out:
        for row in records:
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def initialize_relevance_labels(dataset: Path, audit_dir: Path, output: Path) -> dict[str, int]:
    """Creates an unreviewed template; never derives gold from source matches."""
    if output.exists():
        raise ValueError(f"Refusing to overwrite existing labels: {output}")
    records = load_dataset(dataset)
    verified_audit(dataset, audit_dir)
    candidates = read_jsonl(audit_dir / "anchor_candidates.jsonl")
    indexed = {q["question_id"]: q for q in candidates}
    if len(indexed) != len(records) or set(indexed) != {r["question_id"] for r in records}:
        raise ValueError("Audit candidate question IDs mismatch dataset")
    digest = sha256(dataset)
    template = []
    for row in records:
        source_candidates = []
        for item in indexed[row["question_id"]].get("evidence_snippets", []):
            source_candidates.extend(item.get("chunk_ids", []))
        template.append({
            "question_id": row["question_id"],
            "split": row["split"],
            "answerable": row["answerable"],
            "source_dataset_sha256": digest,
            "candidate_chunk_ids_unverified": list(dict.fromkeys(source_candidates)),
            "verified_relevant_chunk_ids": [],
            "review_state": "unreviewed",
            "reviewer": None,
            "review_notes": "",
        })
    _write_jsonl_new(output, template)
    return {"records": len(template), "unreviewed": len(template)}


def score_retrieval(
    dataset: Path, labels_path: Path, checkpoint_path: Path, split: str, ks: list[int]
) -> dict[str, Any]:
    """Recall@K/MRR@K ONLY on explicitly human-reviewed answerable labels.

    Expected checkpoint rows: {question_id, retrieved_chunk_ids: [ordered IDs]}.
    Expected label rows are produced by initialize_relevance_labels; a reviewer
    must fill verified_relevant_chunk_ids, reviewer, review_state='verified'.
    """
    if split not in ALLOWED_SPLITS:
        raise ValueError("Split must be dev or test")
    if not ks or any(type(k) is not int or k < 1 for k in ks):
        raise ValueError("K must be positive integers")
    questions = load_dataset(dataset)
    index = {q["question_id"]: q for q in questions}
    labels = read_jsonl(labels_path)
    checkpoint = read_jsonl(checkpoint_path)
    if len({r.get("question_id") for r in labels}) != len(labels):
        raise ValueError("Duplicate question ID in relevance labels")
    if len({r.get("question_id") for r in checkpoint}) != len(checkpoint):
        raise ValueError("Duplicate question ID in retrieval checkpoint")
    if any(r.get("question_id") not in index for r in labels + checkpoint):
        raise ValueError("Unknown question ID in labels or retrieval checkpoint")
    label_map = {r["question_id"]: r for r in labels}
    retrieval_map = {r["question_id"]: r for r in checkpoint}
    digest = sha256(dataset)
    rows_scored = []
    skipped = collections.Counter()
    for row in questions:
        if row["split"] != split:
            continue
        if not row["answerable"]:
            skipped["unanswerable_not_scored_as_retrieval_failure"] += 1
            continue
        qid = row["question_id"]
        label = label_map.get(qid)
        if label is None:
            skipped["no_review_label"] += 1
            continue
        if label.get("source_dataset_sha256") != digest or label.get("split") != split or label.get("answerable") is not True:
            raise ValueError(f"{qid}: stale, cross-split, or contradictory labels")
        if label.get("review_state") != "verified":
            skipped["not_human_verified"] += 1
            continue
        reviewer = label.get("reviewer")
        relevant = label.get("verified_relevant_chunk_ids")
        if (not isinstance(reviewer, str) or not reviewer.strip() or
            not isinstance(relevant, list) or not relevant or
            any(not isinstance(cid, str) or not cid.strip() for cid in relevant) or
            len(set(relevant)) != len(relevant)):
            raise ValueError(f"{qid}: verified labels require reviewer and nonempty unique chunk IDs")
        retrieved = retrieval_map.get(qid)
        if retrieved is None:
            skipped["missing_retrieval_checkpoint"] += 1
            continue
        if (retrieved.get("source_dataset_sha256", digest) != digest
            or retrieved.get("split", split) != split):
            raise ValueError(f"{qid}: stale or cross-split retrieval checkpoint")
        if "question_sha256" in retrieved:
            from src.evaluation.benchmark120_hybrid_capture import _question_digest
            if retrieved["question_sha256"] != _question_digest(row["question"]):
                raise ValueError(f"{qid}: retrieval question digest mismatch")
        ranked = retrieved.get("retrieved_chunk_ids")
        if not isinstance(ranked, list) or any(not isinstance(cid, str) or not cid.strip() for cid in ranked):
            raise ValueError(f"{qid}: malformed retrieved_chunk_ids")
        if len(set(ranked)) != len(ranked):
            raise ValueError(f"{qid}: duplicate retrieved chunk IDs")
        truth = set(relevant)
        scores = {}
        for k in ks:
            hits = set(ranked[:k]) & truth
            scores[f"recall@{k}"] = len(hits) / len(truth)
            rank = next((i + 1 for i, cid in enumerate(ranked[:k]) if cid in truth), None)
            scores[f"mrr@{k}"] = 1 / rank if rank else 0.0
        rows_scored.append({"question_id": qid, **scores})
    means = {f"{metric}@{k}": round(sum(r[f"{metric}@{k}"] for r in rows_scored) / len(rows_scored), 6)
             for metric in ("recall", "mrr") for k in ks} if rows_scored else {}
    return {
        "dataset_sha256": digest,
        "split": split,
        "scored_answerable_questions": len(rows_scored),
        "expected_answerable_questions": sum(q["answerable"] and q["split"] == split for q in questions),
        "skipped": dict(sorted(skipped.items())),
        "metrics": means,
        "per_question": rows_scored,
        "warning": "Partial reviewed labels are not a complete benchmark. Unanswerable behavior needs separate answer evaluation.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status", help="Dataset structure only, no correctness claims")
    status.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    check = sub.add_parser("audit", help="Build offline source candidates and human-review worksheets")
    check.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    check.add_argument("--chunks", type=Path, required=True)
    check.add_argument("--out", type=Path, required=True)
    packet = sub.add_parser("export-review-packet", help="Offline: source-locked top-K retrieval excerpts and blank human labels")
    packet.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    packet.add_argument("--chunks", type=Path, required=True)
    packet.add_argument("--checkpoint", type=Path,
                        default=Path("data/processed/benchmark120_hybrid_dev.jsonl"))
    packet.add_argument("--out-dir", type=Path, required=True)
    packet.add_argument("--start", type=int, default=0)
    packet.add_argument("--limit", type=int, default=5)
    packet.add_argument("--top-k", type=int, default=10)
    init = sub.add_parser("init-labels", help="Create empty, unreviewed relevance label template")
    init.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    init.add_argument("--audit-dir", type=Path, required=True)
    init.add_argument("--out", type=Path, required=True)
    capture = sub.add_parser("capture-hybrid", help="Strict hybrid retrieval; defaults to zero live calls")
    capture.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    capture.add_argument("--chunks", type=Path, required=True)
    capture.add_argument("--checkpoint", type=Path, required=True)
    capture.add_argument("--split", choices=sorted(ALLOWED_SPLITS), default="dev")
    capture.add_argument("--max-new", type=int, default=0,
                         help="Maximum NEW Gemini query embeddings (default 0: read-only preview)")
    capture.add_argument("--confirm-live", action="store_true",
                         help="Explicitly authorize live query embeddings for --max-new > 0")
    capture.add_argument("--unlock-test", action="store_true",
                         help="Explicitly authorize running the held-out test split")
    capture.add_argument("--dense-limit", type=int, default=20)
    capture.add_argument("--sparse-limit", type=int, default=20)
    capture.add_argument("--fused-limit", type=int, default=20)
    score = sub.add_parser("score-retrieval", help="Score only verified relevance labels; never create them")
    score.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    score.add_argument("--labels", type=Path, required=True)
    score.add_argument("--checkpoint", type=Path, required=True)
    score.add_argument("--split", choices=sorted(ALLOWED_SPLITS), default="dev")
    score.add_argument("--k", type=int, action="append", default=None)
    score.add_argument("--out", type=Path, help="Optional new JSON report file (no overwrite)")
    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            result = summary(args.dataset)
        elif args.command == "audit":
            load_dataset(args.dataset)
            result = audit(args.dataset, args.chunks, args.out)
        elif args.command == "init-labels":
            result = initialize_relevance_labels(args.dataset, args.audit_dir, args.out)
        elif args.command == "export-review-packet":
            from src.evaluation.benchmark120_retrieval_review import export_review_packet
            result = export_review_packet(
                dataset_path=args.dataset, corpus_path=args.chunks,
                retrieval_path=args.checkpoint, output_dir=args.out_dir,
                start=args.start, limit=args.limit, top_k=args.top_k,
            )
        elif args.command == "capture-hybrid":
            from src.evaluation.benchmark120_hybrid_capture import capture_hybrid
            result = capture_hybrid(
                dataset_path=args.dataset, corpus_path=args.chunks, checkpoint_path=args.checkpoint,
                questions=load_dataset(args.dataset), split=args.split, max_new=args.max_new,
                confirm_live=args.confirm_live, allow_test=args.unlock_test,
                dense_limit=args.dense_limit, sparse_limit=args.sparse_limit, fused_limit=args.fused_limit,
            )
        else:
            result = score_retrieval(args.dataset, args.labels, args.checkpoint, args.split, sorted(set(args.k or [5, 10])))
            if args.out:
                with args.out.open("x", encoding="utf-8") as output:
                    json.dump(result, output, indent=2)
                    output.write("\n")
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
