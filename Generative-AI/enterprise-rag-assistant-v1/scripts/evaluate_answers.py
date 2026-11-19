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
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Case line {line_number} is invalid JSON") from exc
        if not isinstance(case, dict):
            raise ValueError(f"Case line {line_number} is not an object")
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
            review_dir: Path = DEFAULT_REVIEWS, max_new: int = 1, *,
            checkpoint_path: Path = Path("data/processed/hybrid_retrieval_live.jsonl"),
            chunks_path: Path = Path("data/processed/chunks.jsonl")) -> None:
    """Checkpoint each missing retrieval before generation; reuse saved work."""
    if max_new < 1:
        raise ValueError("--max-new must be positive")
    gold, cases, _reviews = prepare(gold_path, cases_path, review_dir)
    pending = [g for g in gold if cases[g["question_id"]]["run"] is None]
    if not pending:
        print("CAPTURE COMPLETE: all answers already saved.")
        return

    from scripts.evaluate_hybrid_retrieval_resumable import run as retrieve_pending
    from src.evaluation.replay import load_replay_inputs

    checkpoints, _chunks = load_replay_inputs(
        gold=gold, checkpoint_path=checkpoint_path, chunks_path=chunks_path, allow_empty=True,
    )
    print(f"BUDGET: at most {max_new} new retrievals and {max_new} generations; one attempt each.")
    for record in pending[:max_new]:
        question_id = record["question_id"]
        if question_id not in checkpoints:
            try:
                retrieve_pending(
                    output=checkpoint_path, max_new=1, gold_path=gold_path,
                    chunks_path=chunks_path, question_ids={question_id},
                )
            except genai_errors.ClientError as exc:
                if getattr(exc, "code", None) != 429:
                    raise
                print("STOPPED: embedding quota exhausted; checkpoints and answers preserved.")
                raise SystemExit(75) from None
            except genai_errors.ServerError as exc:
                print(f"STOPPED: embedding service HTTP {exc.code}; checkpoints preserved.")
                raise SystemExit(75) from None
        # Replay revalidates the file committed by retrieval before spending a
        # generation request. Failure leaves that checkpoint available next run.
        capture_replay(
            gold_path=gold_path, cases_path=cases_path, review_dir=review_dir,
            checkpoint_path=checkpoint_path, chunks_path=chunks_path,
            max_new=1, question_ids={question_id},
        )


def preview_replay(
    question_id: str = "Q001",
    *,
    gold_path: Path = DEFAULT_GOLD_PATH,
    checkpoint_path: Path = Path("data/processed/hybrid_retrieval_live.jsonl"),
    chunks_path: Path = Path("data/processed/chunks.jsonl"),
) -> None:
    """Offline proof that old ranked retrieval can be replayed without an API."""
    from src.evaluation.replay import load_replay_inputs, replay_candidates
    from src.retrieval.context_builder import ContextBuilder

    gold = load_gold_questions(gold_path)
    checkpoints, chunks = load_replay_inputs(
        gold=gold, checkpoint_path=checkpoint_path, chunks_path=chunks_path
    )
    candidates = replay_candidates(question_id, checkpoints, chunks)
    context = ContextBuilder(max_chunks=8, max_context_tokens=4000).build(candidates[:8])
    print("OFFLINE HYBRID REPLAY PREVIEW — NO GEMINI CALLS")
    print(f"QUESTION: {question_id} | SAVED RANKED CHUNKS: {len(candidates)}")
    print(f"SELECTED EVIDENCE: {context['evidence_count']} | TOKEN ESTIMATE: {context['total_tokens']}")
    for item in context["evidence"]:
        print(f"  [Evidence {item['evidence_id']}] {item['chunk_id']} "
              f"(document {item['document_id']}, page {item['page_start']})")
    print("RETRIEVAL SOURCE: saved_hybrid_checkpoint (historical; not a fresh ranking)")
    if context["context"]:
        print("No generated answer. Live capture in replay mode will use generation only.")
    else:
        print("EMPTY EVIDENCE: capture will save an insufficiency response without a generation request.")


def capture_replay(
    *,
    gold_path: Path = DEFAULT_GOLD_PATH,
    cases_path: Path = DEFAULT_CASES,
    review_dir: Path = DEFAULT_REVIEWS,
    checkpoint_path: Path = Path("data/processed/hybrid_retrieval_live.jsonl"),
    chunks_path: Path = Path("data/processed/chunks.jsonl"),
    max_new: int = 1,
    question_ids: set[str] | None = None,
) -> None:
    """Use saved real hybrid rankings, avoiding another query embedding call.

    This is a generation-quality experiment against historic hybrid results,
    NOT a fresh current end-to-end retrieval benchmark.
    """
    if max_new < 1:
        raise ValueError("--max-new must be positive")
    from src.evaluation.replay import is_replayable, load_replay_inputs, replay_candidates
    from src.generation.answer_generator import GroundedAnswerGenerator
    from src.generation.citation_builder import CitationBuilder
    from src.rag.answer_composer import compose_answer
    from src.retrieval.context_builder import ContextBuilder

    gold, cases, _reviews = prepare(gold_path, cases_path, review_dir)
    checkpoints, chunks = load_replay_inputs(
        gold=gold, checkpoint_path=checkpoint_path, chunks_path=chunks_path
    )
    pending = [
        item for item in gold
        if cases[item["question_id"]]["run"] is None
        and (question_ids is None or item["question_id"] in question_ids)
        and item["question_id"] in checkpoints
        and is_replayable(checkpoints[item["question_id"]])
    ]
    if not pending:
        print(
            "NO PENDING SAVED HYBRID RESULTS. "
            "Run the offline status command to distinguish missing retrieval, legacy skips, "
            "and already captured answers."
        )
        return

    # Does not initialize HybridRetriever, GeminiEmbeddingClient, or Qdrant.
    # Only the generation model is constructed, and only when needed.
    generator = None
    context_builder = ContextBuilder(max_chunks=8, max_context_tokens=4000)
    citation_builder = CitationBuilder()

    for record in pending[:max_new]:
        question_id = record["question_id"]
        candidates = replay_candidates(question_id, checkpoints, chunks)
        if generator is None and context_builder.build(candidates[:8])["context"]:
            generator = GroundedAnswerGenerator()
        print(
            f"GENERATING: {question_id} from {len(candidates)} saved hybrid chunks "
            "(ZERO new embedding calls)", flush=True
        )
        try:
            response = compose_answer(
                question=record["question"],
                candidates=candidates,
                retrieval_mode="hybrid",
                retrieval_source="saved_hybrid_checkpoint",
                context_builder=context_builder,
                generator=generator,
                citation_builder=citation_builder,
                retrieval_seconds=None,
                evaluation_trace=True,
            )
        except genai_errors.ClientError as exc:
            if getattr(exc, "code", None) == 429:
                print("STOPPED: Gemini GENERATION API quota exhausted. No embeddings requested.")
                raise SystemExit(75) from None
            raise
        except RuntimeError as exc:
            print(f"STOPPED: Gemini generation failed: {exc}. No embeddings requested.")
            raise SystemExit(75) from None

        cases[question_id] = attach_capture(cases[question_id], response)
        _save_cases(cases_path, gold, cases)
        print(f"SAVED ANSWER: {question_id}. No human quality ratings assigned.", flush=True)


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


def status(*, chunks_path: Path,
           checkpoint_path: Path = Path("data/processed/hybrid_retrieval_live.jsonl"),
           output: Path | None = None, review_packet: Path | None = None,
           retrieval_blocker: str | None = None) -> dict[str, Any]:
    """Validate all saved work and show every question without model calls."""
    from src.evaluation.progress import build_progress, format_review_packet
    from src.evaluation.replay import load_replay_inputs

    # A typo in a report destination must not replace source evaluation data.
    protected = {p.resolve() for p in (
        DEFAULT_GOLD_PATH, DEFAULT_CASES, checkpoint_path, chunks_path,
    )}
    destinations = [p.resolve() for p in (output, review_packet) if p is not None]
    if len(destinations) != len(set(destinations)):
        raise ValueError("Status and review packet must have different destinations")
    for destination in destinations:
        if destination in protected or destination.is_relative_to(DEFAULT_REVIEWS.resolve()):
            raise ValueError("Report destination would overwrite protected evaluation data")
    gold, cases, reviews = prepare()
    checkpoints, chunks = load_replay_inputs(
        gold=gold, checkpoint_path=checkpoint_path, chunks_path=chunks_path, allow_empty=True,
    )
    progress = build_progress(gold, cases, reviews, checkpoints, chunks,
                              retrieval_blocker=retrieval_blocker)
    print("ID    TYPE             ANSWERABLE EXPECTED RETRIEVED ANSWER REVIEW STATUS")
    for row in progress["questions"]:
        print(f"{row['question_id']} {row['question_type']:<16} {str(row['gold_answerable']):<10} "
              f"{row['expected_chunk_count']:<8} {str(row['retrieved_chunk_count']):<9} "
              f"{str(row['answer_status'] == 'GENERATION COMPLETE'):<6} "
              f"{row['human_review']:<10} {row['status']}")
    print("No factual accuracy inferred. Missing expected IDs are diagnostics, not semantic fact scores.")
    if output:
        _write_atomically(output, json.dumps(progress, ensure_ascii=False, indent=2) + "\n")
        print(f"STATUS FILE: {output}")
    if review_packet:
        _write_atomically(review_packet, format_review_packet(gold, cases, checkpoints, chunks))
        print(f"REVIEW PACKET: {review_packet}")
    return progress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="Create gold-anchored cases and blank review forms, offline")
    capture_parser = commands.add_parser(
        "capture", help="LIVE: fresh retrieval or explicitly replay saved hybrid rankings"
    )
    capture_parser.add_argument("--max-new", type=int, default=1)
    capture_parser.add_argument("--replay-hybrid-checkpoint", action="store_true",
                                help="Reuse saved strict-hybrid rankings; no query embeddings")
    capture_parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"),
                                help="Exact SHA-256-locked corpus for replay")
    capture_parser.add_argument("--retrieval-checkpoint", type=Path,
                                default=Path("data/processed/hybrid_retrieval_live.jsonl"))
    preview_parser = commands.add_parser(
        "preview-replay", help="OFFLINE: verify saved hybrid IDs and selected context"
    )
    preview_parser.add_argument("--question-id", default="Q001")
    preview_parser.add_argument("--chunks", type=Path, required=True)
    preview_parser.add_argument("--retrieval-checkpoint", type=Path,
                                default=Path("data/processed/hybrid_retrieval_live.jsonl"))
    inspect_parser = commands.add_parser("inspect", help="Inspect one answer, evidence and gold facts, offline")
    inspect_parser.add_argument("question_id", type=str)
    commands.add_parser("report", help="Summarize real human review only, offline")
    status_parser = commands.add_parser("status", help="OFFLINE: every question, integrity and review state")
    status_parser.add_argument("--chunks", type=Path, required=True)
    status_parser.add_argument("--retrieval-checkpoint", type=Path,
                               default=Path("data/processed/hybrid_retrieval_live.jsonl"))
    status_parser.add_argument("--output", type=Path)
    status_parser.add_argument("--review-packet", type=Path)
    status_parser.add_argument("--retrieval-blocker", type=str,
                               help="Record an observed blocker for missing retrieval; does not claim each question was attempted")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "capture":
        if args.replay_hybrid_checkpoint:
            capture_replay(
                max_new=args.max_new,
                chunks_path=args.chunks,
                checkpoint_path=args.retrieval_checkpoint,
            )
        else:
            capture(max_new=args.max_new, chunks_path=args.chunks,
                    checkpoint_path=args.retrieval_checkpoint)
    elif args.command == "preview-replay":
        preview_replay(
            question_id=args.question_id,
            chunks_path=args.chunks,
            checkpoint_path=args.retrieval_checkpoint,
        )
    elif args.command == "inspect":
        inspect(args.question_id)
    elif args.command == "status":
        status(chunks_path=args.chunks, checkpoint_path=args.retrieval_checkpoint,
               output=args.output, review_packet=args.review_packet,
               retrieval_blocker=args.retrieval_blocker)
    else:
        report()


if __name__ == "__main__":
    main()
