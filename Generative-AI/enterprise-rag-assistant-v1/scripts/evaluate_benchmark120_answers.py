"""RAG-120 answer evaluation from source-verified saved hybrid rankings.

All preview/report calls run offline. CAPTURE invokes Gemini generation only after
--confirm-generation, never embeddings or Qdrant. Test split is locked by default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.benchmark120_answer_replay import report, run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--dataset", type=Path, default=Path("data/evaluation/rag_eval_120.jsonl"))
        p.add_argument("--chunks", type=Path, required=True)
        p.add_argument("--retrieval-checkpoint", type=Path,
                       default=Path("data/processed/benchmark120_hybrid_dev.jsonl"))
        p.add_argument("--answers", type=Path,
                       default=Path("data/processed/benchmark120_answers_dev.jsonl"))
        p.add_argument("--split", choices=("dev", "test"), default="dev")
        p.add_argument("--unlock-test", action="store_true", help="Explicitly unlock held-out test")
    preview = sub.add_parser("preview", help="Verify checkpoint and summarize pending answer generation, offline")
    common(preview)
    capture = sub.add_parser("capture", help="Bounded, resumable Gemini generation only; live consent required")
    common(capture)
    capture.add_argument("--max-new", type=int, default=0)
    capture.add_argument("--confirm-generation", action="store_true")
    capture.add_argument("--model", help="Optional generation model override; do not mix models in one answer checkpoint")
    inspect = sub.add_parser("report", help="Summarize saved answers, optionally export blank human reviews")
    common(inspect)
    inspect.add_argument("--review-export", type=Path)
    args = parser.parse_args(argv)
    kwargs = dict(dataset_path=args.dataset, corpus_path=args.chunks,
                  retrieval_path=args.retrieval_checkpoint, answers_path=args.answers,
                  split=args.split, unlock_test=args.unlock_test)
    if args.command == "preview":
        result = run(**kwargs)
    elif args.command == "capture":
        result = run(**kwargs, max_new=args.max_new, confirm_generation=args.confirm_generation,
                     model_name=args.model)
    else:
        result = report(**kwargs, review_export=args.review_export)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
