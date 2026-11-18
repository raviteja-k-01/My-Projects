"""Answer-quality evaluation: offline preparation, controlled capture, human review.

Windows/WSL project-root commands:
  python -m scripts.evaluate_answers prepare       # offline, no API calls
  python -m scripts.evaluate_answers inspect Q001  # offline, show gold and evidence
  python -m scripts.evaluate_answers capture --max-new 1  # LIVE: 1 Gemini answer
  python -m scripts.evaluate_answers report         # offline, no fake scores

Only the explicitly requested 'capture' command calls Gemini/Qdrant.
Human reviews are saved in individual editable JSON files; the script never
overwrites existing review files, even on interrupted runs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from google.genai import errors as genai_errors

from src.evaluation.answer_quality import (
    attach_capture,
    audit_citation_mechanics,
    new_case,
    new_review,
    summarize_reviews,
    validate_review,
)
from src.evaluation.loader import DEFAULT_GOLD_PATH, load_gold_questions

DEFAULT_CASES = Path("data/processed/answer_eval_cases.jsonl")
DEFAULT_REVIEWS = Path("data/processed/answer_review_forms")


def _write_atomically(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _read_cases(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cases: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        case = json.loads(line)
        question_id = case.get("question_id")
        if not isinstance(question_id, str) or question_id in cases:
            raise ValueError(f"Duplicate/invalid case on line {line_number}")
        cases[question_id] = case
    return cases


def _save_cases(path: Path, gold: list[dict[str, Any]], cases: dict[str, dict[str, Any]]) -> None:
    content = "".join(
        json.dumps(cases[record["question_id"]], ensure_ascii=False) + "\n"
        for record in gold
    )
    _write_atomically(path, content)


def _review_path(root: Path, question_id: str) -> Path:
    # IDs come from a controlled dataset, never a path supplied by a browser.
    if not question_id or not all(ch.isascii() and (ch.isalnum() or ch in "-_") for ch in question_id):
        raise ValueError(f"Unsafe question ID: {question_id!r}")
    return root / f"{question_id}.json"


def _validate_existing_capture(case: dict[str, Any]) -> None:
    run = case.get("run")
    if run is None:
        if case.get("mechanical_checks") is not None:
            raise ValueError("Uncaptured case cannot have mechanical checks")
        return
    if run.get("retrieval_mode") != "hybrid":
        raise ValueError("Mixed BM25 fallback in strict hybrid answer evaluation")
    actual = audit_citation_mechanics(
        run["answer"], run["citations"], run["selected_evidence"]
    )
    if actual != case.get("mechanical_checks"):
        raise ValueError("Stored answer/citations/trace do not match mechanical checks")


def prepare(gold_path: Path = DEFAULT_GOLD_PATH, cases_path: Path = DEFAULT_CASES,
            review_dir: Path = DEFAULT_REVIEWS) -> tuple[
                list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]
            ]:
    """Create independent review forms without changing completed human work."""
    gold = load_gold_questions(gold_path)
    by_id = {g["question_id"]: g for g in gold}
    cases = _read_cases(cases_path)
    unexpected = set(cases) - set(by_id)
    if unexpected:
        raise ValueError(f"Unrecognized existing captured cases: {sorted(unexpected)}")

    changed = False
    for record in gold:
        question_id = record["question_id"]
        expected_case = new_case(record)
        if question_id not in cases:
            cases[question_id] = expected_case
            changed = True
        else:
            stored = cases[question_id]
            for field, expected in expected_case.items():
                if field in ("run", "mechanical_checks"):
                    continue
                if stored.get(field) != expected:
                    raise ValueError(
                        f"Gold dataset changed at {question_id} / {field}. "
                        "Don't silently reuse a stale answer capture."
                    )
            _validate_existing_capture(stored)

    if changed:
        _save_cases(cases_path, gold, cases)

    reviews = {}
    review_dir.mkdir(parents=True, exist_ok=True)
    for record in gold:
        question_id = record["question_id"]
        review_file = _review_path(review_dir, question_id)
        if review_file.exists():
            review = json.loads(review_file.read_text(encoding="utf-8"))
        else:
            review = new_review(record)
            _write_atomically(
                review_file, json.dumps(review, ensure_ascii=False, indent=2) + "\n"
            )
        validate_review(review, record)
        reviews[question_id] = review

    print(
        f"GOLD: {len(gold)} | ANSWERABLE: {sum(g['answerable'] for g in gold)} "
        f"| UNANSWERABLE: {sum(not g['answerable'] for g in gold)}"
    )
    print(
        f"EXPECTED FACTS: {sum(len(g['expected_facts']) for g in gold)} "
        f"| CAPTURED ANSWERS: {sum(cases[g['question_id']]['run'] is not None for g in gold)}"
    )
    print(f"REVIEW FORMS: {review_dir} | No model scores assigned.")
    return gold, cases, reviews


def capture(gold_path: Path = DEFAULT_GOLD_PATH, cases_path: Path = DEFAULT_CASES,
            review_dir: Path = DEFAULT_REVIEWS, max_new: int = 1) -> None:
    if max_new < 1:
        raise ValueError("--max-new must be positive")
    gold, cases, _reviews = prepare(gold_path, cases_path, review_dir)
    pending = [g for g in gold if cases[g["question_id"]]["run"] is None]
    if not pending:
        print("CAPTURE COMPLETE: all answers already saved.")
        return

    # Import/initialize expensive remote resources only for an explicit live run.
    from src.rag.rag_pipeline import EnterpriseRAGPipeline

    pipeline = EnterpriseRAGPipeline()
    captured = 0
    try:
        for record in pending[:max_new]:
            question_id = record["question_id"]
            print(f"CAPTURING: {question_id} (hybrid only)", flush=True)
            try:
                result = pipeline.answer(
                    record["question"],
                    evaluation_trace=True,
                    allow_sparse_fallback=False,
                )
            except genai_errors.ClientError as exc:
                if getattr(exc, "code", None) == 429:
                    print("STOPPED: Gemini quota exhausted. Previous answers saved.")
                    raise SystemExit(75) from None
                raise
            except RuntimeError as exc:
                # Generation client wraps some 429 errors into RuntimeError.
                print(f"STOPPED: generation failed; saved answers preserved: {exc}")
                raise SystemExit(75) from None

            cases[question_id] = attach_capture(cases[question_id], result)
            _save_cases(cases_path, gold, cases)
            captured += 1
            print(f"SAVED: {question_id} | total: "
                  f"{sum(cases[g['question_id']]['run'] is not None for g in gold)}/{len(gold)}",
                  flush=True)
    finally:
        pipeline.close()
    print(f"NEW CAPTURES: {captured} | No accuracy scores inferred.")


def inspect(question_id: str, gold_path: Path = DEFAULT_GOLD_PATH,
            cases_path: Path = DEFAULT_CASES, review_dir: Path = DEFAULT_REVIEWS) -> None:
    gold, cases, reviews = prepare(gold_path, cases_path, review_dir)
    by_id = {g["question_id"]: g for g in gold}
    if question_id not in by_id:
        raise ValueError(f"Unknown gold question: {question_id}")
    row = by_id[question_id]
    case = cases[question_id]
    print(f"\n{question_id}: {row['question']}")
    print(f"TYPE: {row['question_type']} | ANSWERABLE: {row['answerable']}")
    print("EXPECTED FACTS (review each against corpus and generated answer):")
    for i, fact in enumerate(row["expected_facts"], start=1):
        print(f"  {i}. {fact}")
    run = case["run"]
    if run is None:
        print("ANSWER: NOT CAPTURED. Do not score this question yet.")
        return
    print(f"\nANSWER:\n{run['answer']}")
    print(f"\nCITATION MECHANICS: {json.dumps(case['mechanical_checks'], indent=2)}")
    print("\nACTUAL EVIDENCE SUPPLIED TO THE MODEL:")
    for e in run["selected_evidence"]:
        print(f"\n[Evidence {e['evidence_id']}] {e['chunk_id']} "
              f"(document {e.get('document_id')}, page {e.get('page_start')})")
        print(e["text"])
    print(f"\nREVIEW FILE: {_review_path(review_dir, question_id)}")
    print("Read each claim against its cited text; fill ratings 0/1/2 and fact statuses.")
    print(f"REVIEW COMPLETE: {validate_review(reviews[question_id], row)}")


def report(gold_path: Path = DEFAULT_GOLD_PATH, cases_path: Path = DEFAULT_CASES,
           review_dir: Path = DEFAULT_REVIEWS) -> dict[str, Any]:
    gold, cases, reviews = prepare(gold_path, cases_path, review_dir)
    summary = summarize_reviews(gold, cases, reviews)
    print("\nANSWER QUALITY AUDIT — MANUAL REVIEW ONLY")
    print(f"CAPTURED: {summary['captured']}/{summary['gold_questions']}")
    print(f"FULLY REVIEWED: {summary['fully_human_reviewed']}/{summary['gold_questions']}")
    print(f"MECHANICAL CITATION FAILURES: {summary['mechanical_citation_failures']}")
    print(f"EXPECTED FACT STATUS COUNTS: {summary['expected_fact_statuses']}")
    print(f"FLAGGED REVIEWED CASES: {summary['reviewed_problem_cases']}")
    for dimension, counts in summary["ratings_distribution"].items():
        print(f"{dimension}: 0={counts.get(0, 0)}, 1={counts.get(1, 0)}, 2={counts.get(2, 0)}")
    print(summary["note"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="Create gold-anchored cases and blank review forms, offline")
    capture_parser = commands.add_parser("capture", help="LIVE: query RAG and save answers one by one")
    capture_parser.add_argument("--max-new", type=int, default=1)
    inspect_parser = commands.add_parser("inspect", help="Inspect one answer, evidence and gold facts, offline")
    inspect_parser.add_argument("question_id", type=str)
    commands.add_parser("report", help="Summarize real human review only, offline")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "capture":
        capture(max_new=args.max_new)
    elif args.command == "inspect":
        inspect(args.question_id)
    else:
        report()


if __name__ == "__main__":
    main()
