"""Reconcile RAG-120 retrieval, answer, and review checkpoints without live calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.benchmark120_answer_replay import report
from src.evaluation.benchmark120_evidence import read_jsonl


def count_jsonl(path: Path) -> tuple[int, int]:
    rows = read_jsonl(path)
    ids = [row.get("question_id") for row in rows]
    return len(rows), len(set(qid for qid in ids if isinstance(qid, str)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--groq-answers", type=Path,
                        default=Path("data/processed/benchmark120_answers_dev_groq.jsonl"))
    parser.add_argument("--gemini-answers", type=Path,
                        default=Path("data/processed/benchmark120_answers_dev.jsonl"))
    parser.add_argument("--retrieval", type=Path,
                        default=Path("data/processed/benchmark120_hybrid_dev.jsonl"))
    parser.add_argument("--reviews", type=Path,
                        default=Path("data/processed/benchmark120_groq_review_80.jsonl"))
    args = parser.parse_args()

    retrieval_count, retrieval_unique = count_jsonl(args.retrieval)
    groq_count, groq_unique = count_jsonl(args.groq_answers)
    gemini_count, gemini_unique = count_jsonl(args.gemini_answers)
    review_count, review_unique = count_jsonl(args.reviews)
    reviewed = 0
    if args.reviews.exists():
        reviewed = sum(
            row.get("review", {}).get("review_state") == "complete"
            and row.get("review", {}).get("reviewer_kind") == "human"
            for row in read_jsonl(args.reviews)
        )

    replay = report(
        dataset_path=Path("data/evaluation/rag_eval_120.jsonl"),
        corpus_path=args.chunks,
        retrieval_path=args.retrieval,
        answers_path=args.groq_answers,
        split="dev",
        reviews_path=args.reviews,
    )
    print(json.dumps({
        "retrieval": {"rows": retrieval_count, "unique_question_ids": retrieval_unique},
        "groq_answers": {"rows": groq_count, "unique_question_ids": groq_unique},
        "gemini_legacy_answers": {"rows": gemini_count, "unique_question_ids": gemini_unique},
        "groq_reviews": {
            "rows": review_count,
            "unique_question_ids": review_unique,
            "completed_human_reviews": reviewed,
        },
        "groq_replay": replay,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
