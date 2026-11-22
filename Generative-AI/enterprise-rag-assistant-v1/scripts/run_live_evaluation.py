#!/usr/bin/env python3
"""Resumable live 120-question RAG evaluation through the FastAPI /ask endpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from scripts.evaluate_benchmark120 import load_dataset
from src.evaluation.answer_quality import audit_citation_mechanics
from src.evaluation.benchmark120_evidence import sha256


DEFAULT_DATASET = Path("data/evaluation/rag_eval_120.jsonl")
DEFAULT_OUTPUT = Path("data/processed/live_evaluation_120.jsonl")
DEFAULT_WORKBOOK = Path("RAG_Live_Evaluation_Report.xlsx")
DEFAULT_FREEZE = Path("data/processed/live_evaluation_dev_freeze.json")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".live-evaluation-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupt result JSONL at line {line_number}") from exc
        if not isinstance(row, dict) or not isinstance(row.get("question_id"), str):
            raise ValueError(f"Invalid result record at line {line_number}")
        rows.append(row)
    return rows


def append_result(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def append_ledger(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "timestamp_utc": utc_now(),
            **event,
        }, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def ledger_starts(path: Path) -> int:
    if not path.exists():
        return 0
    starts = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupt live-request ledger at line {line_number}") from exc
        if event.get("event") == "started":
            starts += 1
    return starts


def normalize(text: str) -> str:
    import re
    import unicodedata
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold(), re.UNICODE))


def deterministic_fact_metrics(answer: str, expected_facts: list[str]) -> dict[str, Any]:
    normalized_answer = normalize(answer)
    results = []
    met = 0
    for fact in expected_facts:
        fact_text = normalize(fact)
        matched = bool(fact_text) and fact_text in normalized_answer
        met += int(matched)
        results.append({
            "fact": fact,
            "deterministic_text_match": matched,
            "measurement_type": "deterministic_text_match_only",
        })
    return {
        "kind": "deterministic_text_match",
        "fact_count": len(expected_facts),
        "facts_matched": met,
        "facts_unmatched": len(expected_facts) - met,
        "fact_results": results,
        "nota": "Text matching is not factual grounding or human verification.",
    }


def fingerprint_configuration(api_url: str, dataset: Path, split: str) -> dict[str, Any]:
    source_files = (
        "src/api/app.py",
        "src/rag/rag_pipeline.py",
        "src/rag/answer_composer.py",
        "src/retrieval/hybrid_retriever.py",
        "src/retrieval/context_builder.py",
        "src/generation/citation_builder.py",
        "src/generation/answer_generator.py",
    )
    source_hashes = {
        name: sha256(Path(name))
        for name in source_files
        if Path(name).exists()
    }
    return {
        "schema": 1,
        "split": split,
        "api_url": api_url,
        "dataset_sha256": sha256(dataset),
        "dataset_path": str(dataset),
        "source_files_sha256": source_hashes,
        "created_utc": utc_now(),
    }


def hash_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def stable_configuration_hash(configuration: dict[str, Any]) -> str:
    stable = dict(configuration)
    stable.pop("created_utc", None)
    stable.pop("configuration_sha256", None)
    return hash_payload(stable)


def validate_freeze(freeze_path: Path, configuration: dict[str, Any]) -> None:
    if not freeze_path.exists():
        raise ValueError(
            "Held-out test execution is blocked: freeze development first "
            "with --freeze-dev after all 80 development captures complete."
        )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("split") != "dev":
        raise ValueError("Invalid development freeze")
    if freeze.get("configuration_sha256") != stable_configuration_hash(configuration):
        raise ValueError("Development freeze does not match current configuration")
    if freeze.get("dev_question_count") != 80:
        raise ValueError("Development freeze is incomplete")


def get_existing(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_results(path)
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = row["question_id"]
        if qid in seen:
            raise ValueError(f"Duplicate question ID in live result checkpoint: {qid}")
        seen[qid] = row
    return seen


def call_api(
    *,
    api_url: str,
    question: str,
    demo_token: str | None,
    retry_rate_limit: bool = False,
    max_rate_limit_retries: int = 0,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    headers = {"Content-Type": "application/json"}
    if demo_token:
        headers["X-RAG-Demo-Token"] = demo_token
    payload = {"question": question, "evaluation_trace": True}
    attempts: list[dict[str, Any]] = []
    for attempt_number in range(max_rate_limit_retries + 1 if retry_rate_limit else 1):
        started = time.perf_counter()
        try:
            response = requests.post(
                api_url.rstrip("/") + "/ask",
                json=payload,
                headers=headers,
                timeout=(10, 300),
            )
            elapsed = time.perf_counter() - started
            try:
                body = response.json()
            except ValueError:
                body = {"detail": "Non-JSON response"}
            if response.status_code != 200:
                attempt = {
                    "http_status": response.status_code,
                    "error_category": "api_http_error",
                    "error_detail": str(body.get("detail", "Request failed")),
                    "latency_seconds": elapsed,
                    "captured": False,
                    "retry_after_seconds": (
                        float(response.headers["Retry-After"])
                        if response.headers.get("Retry-After", "").strip().isdigit()
                        else None
                    ),
                    "ratelimit": {
                        name: response.headers.get(header)
                        for name, header in (
                            ("remaining_requests", "X-Ratelimit-Remaining-Requests"),
                            ("remaining_tokens", "X-Ratelimit-Remaining-Tokens"),
                            ("reset_requests", "X-Ratelimit-Reset-Requests"),
                            ("reset_tokens", "X-Ratelimit-Reset-Tokens"),
                        )
                        if response.headers.get(header)
                    },
                }
                attempts.append(attempt)
                if (
                    retry_rate_limit
                    and attempt_number < max_rate_limit_retries
                    and response.status_code == 503
                    and "429" in str(attempt["error_detail"])
                ):
                    retry_after = attempt["retry_after_seconds"]
                    if retry_after is not None:
                        time.sleep(min(max(retry_after, 1), 86400))
                    continue
                return None, {**attempt, "attempt_count": len(attempts), "attempts": attempts}
            attempt = {
                "http_status": 200,
                "error_category": None,
                "error_detail": None,
                "latency_seconds": elapsed,
                "captured": True,
                "retry_after_seconds": None,
                "ratelimit": {},
            }
            attempts.append(attempt)
            return body, {**attempt, "attempt_count": len(attempts), "attempts": attempts}
        except requests.RequestException as exc:
            elapsed = time.perf_counter() - started
            category = "timeout" if isinstance(exc, requests.Timeout) else "transport_error"
            attempt = {
                "http_status": None,
                "error_category": category,
                "error_detail": type(exc).__name__,
                "latency_seconds": elapsed,
                "captured": False,
                "retry_after_seconds": None,
                "ratelimit": {},
            }
            attempts.append(attempt)
            if not retry_rate_limit or attempt_number >= max_rate_limit_retries:
                return None, {**attempt, "attempt_count": len(attempts), "attempts": attempts}
    return None, {
        "http_status": None,
        "error_category": "transport_error",
        "error_detail": "Request retries exhausted",
        "latency_seconds": 0,
        "captured": False,
        "attempt_count": len(attempts),
        "attempts": attempts,
    }


def build_workbook(
    *,
    dataset: Path,
    results_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError as exc:
        raise RuntimeError("openpyxl is required; install requirements.txt") from exc

    questions = load_dataset(dataset)
    results = get_existing(results_path)
    wb = Workbook()
    wb.remove(wb.active)

    def title(sheet, text: str, subtitle: str | None = None) -> None:
        sheet.sheet_view.showGridLines = False
        sheet.merge_cells("A1:V1")
        cell = sheet["A1"]
        cell.value = text
        cell.font = Font(bold=True, size=16, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.alignment = Alignment(horizontal="left")
        if subtitle:
            sheet.merge_cells("A2:V2")
            sheet["A2"] = subtitle
            sheet["A2"].font = Font(italic=True, color="666666")

    def header(sheet, row: int, names: list[str]) -> None:
        for index, name in enumerate(names, 1):
            cell = sheet.cell(row=row, column=index, value=name)
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="5B9BD5")
            cell.alignment = Alignment(horizontal="center", vertical="center")

    def width(sheet) -> None:
        for column in range(1, sheet.max_column + 1):
            values = [str(sheet.cell(row=row, column=column).value or "") for row in range(1, sheet.max_row + 1)]
            length = max((len(value) for value in values), default=10)
            sheet.column_dimensions[get_column_letter(column)].width = min(max(length + 2, 12), 55)

    dashboard = wb.create_sheet("Dashboard")
    title(dashboard, "Enterprise RAG | 120-Question Live Evaluation")
    dashboard.append([])
    dashboard.append(["Dataset", str(dataset)])
    dashboard.append(["Questions", len(questions)])
    dashboard.append(["Development", sum(q["split"] == "dev" for q in questions)])
    dashboard.append(["Held-out test", sum(q["split"] == "test" for q in questions)])
    dashboard.append(["Actual API captures", sum(r.get("captured") is True for r in results.values())])
    dashboard.append(["API errors", sum(r.get("captured") is False and r.get("question_id") in results for r in results.values())])
    dashboard.append(["Human reviews complete", 0])
    dashboard.append(["AI judge assessments", 0])
    dashboard.append(["Status", "Live execution pending explicit authorization" if not results else "Partial or complete live capture"])
    dashboard.append(["Actual API captures (formula)", "=COUNTIF('Live Captures'!F4:F123,TRUE)"])
    dashboard.append(["Pending questions (formula)", "=COUNTIF('Live Captures'!G4:G123,\"pending\")"])
    dashboard.append(["Human reviews complete (formula)", "=COUNTIF('Review Metrics'!B4:B123,\"Complete\")"])
    dashboard.append(["Request errors (formula)", "=COUNTA('Errors'!A4:A123)"])

    live = wb.create_sheet("Live Captures")
    title(live, "Live Captures", "Only API /ask responses are counted here. Saved answers are not live captures.")
    live_headers = [
        "Question ID", "Split", "Question", "Answerable", "Expected Facts",
        "Captured?", "Status", "Answer", "Citations", "Selected Evidence",
        "Retrieved Chunk IDs", "Retrieval Mode", "Retrieval Source",
        "Evidence Count", "Model", "HTTP Status", "Error Category",
        "Latency Seconds", "Server Latency Seconds", "Captured UTC",
        "Payload SHA256", "Source Data", "Matched Facts (Deterministic)",
        "Fact Count", "Citation Mechanics", "Groundedness",
        "Human Review State"
    ]
    header(live, 3, live_headers)
    for row_index, question in enumerate(questions, 4):
        record = results.get(question["question_id"])
        source = "actual_live_api" if record and record.get("captured") else "pending"
        response = record.get("response", {}) if record else {}
        trace = response.get("evaluation_trace", {}) if isinstance(response, dict) else {}
        citations = response.get("citations", []) if isinstance(response, dict) else []
        evidence = trace.get("selected_evidence", []) if isinstance(trace, dict) else []
        deterministic = record.get("deterministic_metrics", {}) if record else {}
        fact_match = deterministic.get("fact_match", {}) if isinstance(deterministic, dict) else {}
        citation_mechanics = deterministic.get("citation_mechanics", {}) if isinstance(deterministic, dict) else {}
        values = [
            question["question_id"], question["split"], question["question"], question["answerable"],
            json.dumps(question["expected_facts"], ensure_ascii=False),
            bool(record and record.get("captured")), "captured" if record and record.get("captured") else
            ("error" if record else "pending"),
            response.get("answer") if isinstance(response, dict) else "",
            json.dumps(citations, ensure_ascii=False),
            json.dumps(evidence, ensure_ascii=False),
            json.dumps(trace.get("retrieved_chunk_ids", []), ensure_ascii=False),
            response.get("retrieval_mode", "") if isinstance(response, dict) else "",
            response.get("retrieval_source", "") if isinstance(response, dict) else "",
            response.get("evidence_count", "") if isinstance(response, dict) else "",
            response.get("model", "") if isinstance(response, dict) else "",
            record.get("http_status", "") if record else "",
            record.get("error_category", "") if record else "",
            record.get("latency_seconds", "") if record else "",
            response.get("latency", {}).get("total_seconds", "") if isinstance(response, dict) else "",
            record.get("captured_utc", "") if record else "",
            record.get("payload_sha256", "") if record else "",
            source,
            fact_match.get("facts_matched", "") if fact_match else "",
            fact_match.get("fact_count", "") if fact_match else "",
            "Pass" if citation_mechanics.get("mechanically_valid") is True else
            ("Fail" if citation_mechanics else ""),
            deterministic.get("groundedness", "Human or AI judge required") if deterministic else "",
            "Pending",
        ]
        for column, value in enumerate(values, 1):
            live.cell(row=row_index, column=column, value=value)
    live.freeze_panes = "A4"
    live.auto_filter.ref = live.dimensions
    width(live)

    review = wb.create_sheet("Review Metrics")
    title(review, "Human Review Metrics", "Official metrics remain blank until completed and signed human reviews exist.")
    review_headers = [
        "Question ID", "Human Review State", "Reviewer", "Required Facts Reviewed",
        "Facts Fully Met", "Facts Partially Met", "Claims Reviewed", "Claims Supported",
        "Claims Cited", "Citation Pairs Checked", "Citation Pairs Correct",
        "Abstention Correct?", "Final Verdict", "Fact Completeness",
        "Groundedness", "Citation Coverage", "Citation Correctness", "Notes"
    ]
    header(review, 3, review_headers)
    for row_index, question in enumerate(questions, 4):
        values = [question["question_id"], "Pending", "", "", "", "", "", "", "", "", "", "", "PENDING",
                  "", "", "", "", ""]
        for column, value in enumerate(values, 1):
            review.cell(row=row_index, column=column, value=value)
    review.freeze_panes = "A4"
    width(review)

    evidence = wb.create_sheet("Evidence")
    title(evidence, "Evidence and Citation Audit")
    evidence_headers = [
        "Question ID", "Evidence ID", "Chunk ID", "Document ID", "Document Name",
        "Page Start", "Page End", "Section", "Content Type", "Text",
        "Citation Present?", "Mechanical Check", "Source Support", "Human Review State"
    ]
    header(evidence, 3, evidence_headers)
    evidence_row = 4
    for question in questions:
        record = results.get(question["question_id"])
        if not record or not record.get("captured"):
            continue
        response = record["response"]
        selected = response.get("evaluation_trace", {}).get("selected_evidence", [])
        cited_ids = {c["evidence_id"] for c in response.get("citations", [])}
        citations = response.get("citations", [])
        mechanically_valid = audit_citation_mechanics(
            response.get("answer", ""), citations, selected
        )["mechanically_valid"]
        for item in selected:
            evidence_id = item["evidence_id"]
            values = [
                question["question_id"], evidence_id, item.get("chunk_id"),
                item.get("document_id"), item.get("document_name"),
                item.get("page_start"), item.get("page_end"),
                item.get("section_label") or item.get("section_title"),
                item.get("content_type"), item.get("text"),
                evidence_id in cited_ids, mechanically_valid,
                "Unverified", "Pending",
            ]
            for column, value in enumerate(values, 1):
                evidence.cell(row=evidence_row, column=column, value=value)
            evidence_row += 1
    width(evidence)

    errors = wb.create_sheet("Errors")
    title(errors, "Request and Pipeline Errors")
    error_headers = ["Question ID", "Split", "HTTP Status", "Error Category", "Error Detail", "Latency Seconds", "UTC"]
    header(errors, 3, error_headers)
    error_row = 4
    for question in questions:
        record = results.get(question["question_id"])
        if record and not record.get("captured"):
            values = [
                question["question_id"], question["split"], record.get("http_status"),
                record.get("error_category"), record.get("error_detail"),
                record.get("latency_seconds"), record.get("captured_utc"),
            ]
            for column, value in enumerate(values, 1):
                errors.cell(row=error_row, column=column, value=value)
            error_row += 1
    width(errors)

    ai_judge = wb.create_sheet("AI Judge")
    title(ai_judge, "AI-Judge Assessments", "No AI-judge result is recorded unless explicitly run and labeled as an assessment.")
    ai_headers = ["Question ID", "Judge Status", "Claim Groundedness", "Scope Fidelity", "Contradictions", "Missing Qualifiers", "Evidence IDs", "Model", "Notes"]
    header(ai_judge, 3, ai_headers)
    for row_index, question in enumerate(questions, 4):
        values = [question["question_id"], "Not run", "", "", "", "", "", "", ""]
        for column, value in enumerate(values, 1):
            ai_judge.cell(row=row_index, column=column, value=value)
    width(ai_judge)

    tracker = Path("RAG_Live_Evaluation_Tracker_v10.xlsx")
    if tracker.exists():
        try:
            from openpyxl import load_workbook
            source_wb = load_workbook(tracker, read_only=False, data_only=False)
            if "Live Tests" in source_wb.sheetnames:
                source = source_wb["Live Tests"]
                imported = wb.create_sheet("Imported Manual Observations")
                title(imported, "Imported Manual Observations",
                      "User-provided screenshots/transcripts only. Not counted as actual API captures.")
                imported_headers = [
                    "Question ID", "Gold Answerable", "Benchmark Question",
                    "Manual Captured?", "Manual Answer", "Displayed Sources",
                    "Retrieval Mode", "Evidence Chunks", "Latency",
                    "Observation Source", "Observation / Limitation"
                ]
                header(imported, 3, imported_headers)
                target_row = 4
                for source_row in range(5, source.max_row + 1):
                    qid = source.cell(source_row, 1).value
                    if not isinstance(qid, str) or not qid.startswith("E"):
                        continue
                    values = [
                        qid,
                        source.cell(source_row, 2).value,
                        source.cell(source_row, 4).value,
                        source.cell(source_row, 5).value,
                        source.cell(source_row, 6).value,
                        source.cell(source_row, 7).value,
                        source.cell(source_row, 8).value,
                        source.cell(source_row, 9).value,
                        source.cell(source_row, 10).value,
                        source.cell(source_row, 11).value,
                        source.cell(source_row, 12).value,
                    ]
                    for column, value in enumerate(values, 1):
                        imported.cell(row=target_row, column=column, value=value)
                    target_row += 1
                width(imported)
        except Exception:
            pass

    protocol = wb.create_sheet("Protocol")
    title(protocol, "Evaluation Protocol")
    protocol_rows = [
        ("Live source", "FastAPI /ask endpoint; no saved benchmark answers"),
        ("Reference answers", "Never sent to the RAG generation request"),
        ("Human metrics", "Blank until independent human review is completed"),
        ("AI judge", "Separate and explicitly labeled; never converted to human review"),
        ("Deterministic matches", "Text matching only; not factual grounding"),
        ("Held-out test", "Requires configuration freeze and explicit --unlock-test"),
        ("Resume", "Only uncaptured questions are requested"),
        ("Saved checkpoint", str(results_path)),
    ]
    for row, (key, value) in enumerate(protocol_rows, 3):
        protocol.cell(row=row, column=1, value=key).font = Font(bold=True)
        protocol.cell(row=row, column=2, value=value)
    width(protocol)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    return {"workbook": str(output_path), "questions": len(questions), "captured": sum(r.get("captured") is True for r in results.values())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--api-url", default=os.getenv("RAG_API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--unlock-test", action="store_true")
    parser.add_argument("--max-new", type=int, default=0)
    parser.add_argument("--confirm-live", action="store_true")
    parser.add_argument("--request-budget", type=int)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--retry-rate-limit", action="store_true")
    parser.add_argument("--max-rate-limit-retries", type=int, default=0)
    parser.add_argument("--demo-token-env", default="RAG_DEMO_API_TOKEN")
    parser.add_argument("--freeze-dev", action="store_true")
    args = parser.parse_args(argv)

    questions = load_dataset(args.dataset)
    subset = [q for q in questions if q["split"] == args.split]
    if args.split == "test" and not args.unlock_test:
        raise ValueError("Held-out test split requires --unlock-test")

    configuration = fingerprint_configuration(args.api_url, args.dataset, args.split)
    configuration["configuration_sha256"] = stable_configuration_hash(configuration)
    if args.split == "test":
        validate_freeze(args.freeze, configuration)

    existing = get_existing(args.output)
    captured = {qid for qid, row in existing.items() if row.get("captured")}
    error_ids = {qid for qid, row in existing.items() if not row.get("captured")}
    pending = [
        q for q in subset
        if q["question_id"] not in captured
        and (args.retry_errors or q["question_id"] not in error_ids)
    ]

    report: dict[str, Any] = {
        "split": args.split,
        "questions": len(subset),
        "captured": len(captured.intersection(q["question_id"] for q in subset)),
        "pending": len(pending),
        "configuration": configuration,
        "output": str(args.output),
        "workbook": str(args.workbook),
        "live_calls_enabled": bool(args.max_new and args.confirm_live),
        "newly_captured": 0,
        "errors": sum(not row.get("captured") for qid, row in existing.items() if qid in {q["question_id"] for q in subset}),
    }
    if args.max_new and not args.confirm_live:
        raise ValueError("Live API requests require --confirm-live")
    if args.request_budget is not None and (type(args.request_budget) is not int or args.request_budget < 0):
        raise ValueError("Request budget must be a nonnegative integer")
    if args.max_new and args.request_budget is None:
        raise ValueError("Live evaluation requires an explicit --request-budget")
    if args.max_new and args.request_budget <= 0:
        raise ValueError("Live evaluation request budget must be positive")
    if args.max_new > len(pending):
        raise ValueError("--max-new cannot exceed the pending uncaptured question count")
    if args.request_budget is not None and args.max_new > args.request_budget:
        raise ValueError("--max-new cannot exceed --request-budget")
    if type(args.max_rate_limit_retries) is not int or not 0 <= args.max_rate_limit_retries <= 3:
        raise ValueError("--max-rate-limit-retries must be between 0 and 3")
    if args.retry_rate_limit and args.max_rate_limit_retries == 0:
        raise ValueError("--retry-rate-limit requires a positive --max-rate-limit-retries")

    demo_token = os.getenv(args.demo_token_env)
    lock = args.output.with_suffix(args.output.suffix + ".lock")
    handle = None
    if args.max_new:
        budget_path = args.output.with_suffix(args.output.suffix + ".budget.json")
        ledger_path = args.output.with_suffix(args.output.suffix + ".attempts.jsonl")
        prior_starts = ledger_starts(ledger_path)
        configuration_hash = configuration["configuration_sha256"]
        if budget_path.exists():
            budget = json.loads(budget_path.read_text(encoding="utf-8"))
            if (budget.get("schema") != 1
                or budget.get("configuration_sha256") != configuration_hash
                or type(budget.get("limit")) is not int):
                raise ValueError("Persistent live request budget is invalid or for another configuration")
            if args.request_budget != budget["limit"]:
                raise ValueError(
                    "Persistent request budget already set; use the exact same "
                    "--request-budget; a budget extension requires a separate new output"
                )
        else:
            atomic_json(budget_path, {
                "schema": 1,
                "configuration_sha256": configuration_hash,
                "limit": args.request_budget,
                "created_utc": utc_now(),
            })
        remaining_budget = args.request_budget - prior_starts
        maximum_attempts = args.max_new * (
            args.max_rate_limit_retries + 1 if args.retry_rate_limit else 1
        )
        if remaining_budget < maximum_attempts:
            raise ValueError(
                f"Request budget remaining={remaining_budget}; "
                f"--max-new={args.max_new} with up to "
                f"{args.max_rate_limit_retries} rate-limit retries would exceed it"
            )
        try:
            handle = lock.open("x", encoding="utf-8")
            handle.write(json.dumps({"pid": os.getpid(), "started_utc": utc_now()}))
            handle.flush()
            os.fsync(handle.fileno())
        except FileExistsError as exc:
            raise ValueError("Live evaluation is locked by another or interrupted run") from exc

    try:
        for question in pending[:args.max_new] if args.max_new else []:
            qid = question["question_id"]
            payload_hash = hash_payload({"question": question["question"], "evaluation_trace": True})
            ledger_path = args.output.with_suffix(args.output.suffix + ".attempts.jsonl")
            response, status = call_api(
                api_url=args.api_url,
                question=question["question"],
                demo_token=demo_token,
                retry_rate_limit=args.retry_rate_limit,
                max_rate_limit_retries=args.max_rate_limit_retries,
            )
            for attempt_number, attempt in enumerate(status.get("attempts", [status]), 1):
                append_ledger(ledger_path, {
                    "event": "started",
                    "question_id": qid,
                    "attempt": attempt_number,
                    "configuration_sha256": configuration["configuration_sha256"],
                })
                append_ledger(ledger_path, {
                    "event": "finished",
                    "question_id": qid,
                    "attempt": attempt_number,
                    "captured": attempt["captured"],
                    "http_status": attempt["http_status"],
                    "error_category": attempt["error_category"],
                    "retry_after_seconds": attempt.get("retry_after_seconds"),
                    "ratelimit": attempt.get("ratelimit", {}),
                    "configuration_sha256": configuration["configuration_sha256"],
                })
            row: dict[str, Any] = {
                "question_id": qid,
                "split": args.split,
                "question": question["question"],
                "answerable": question["answerable"],
                "reference_answer": question["reference_answer"],
                "expected_facts": question["expected_facts"],
                "configuration": configuration,
                "payload_sha256": payload_hash,
                "captured_utc": utc_now(),
                "captured": status["captured"],
                "http_status": status["http_status"],
                "error_category": status["error_category"],
                "error_detail": status["error_detail"],
                "latency_seconds": status["latency_seconds"],
                "response": response,
            }
            if response:
                selected = response.get("evaluation_trace", {}).get("selected_evidence", [])
                citations = response.get("citations", [])
                fact_metrics = deterministic_fact_metrics(
                    response.get("answer", ""), question["expected_facts"]
                )
                citation_mechanics = audit_citation_mechanics(
                    response.get("answer", ""), citations, selected
                )
                row["deterministic_metrics"] = {
                    "fact_match": fact_metrics,
                    "citation_mechanics": citation_mechanics,
                    "abstention_detected": not selected and normalize(response.get("answer", "")).startswith(
                        "there is insufficient evidence"
                    ),
                    "scope_fidelity": "not_assessed",
                    "groundedness": "human_or_ai_judge_required",
                }
            append_result(args.output, row)
            report["newly_captured"] += int(status["captured"])
            report["captured"] = len(captured.intersection(q["question_id"] for q in subset)) + report["newly_captured"]

        if args.freeze_dev:
            if args.split != "dev":
                raise ValueError("--freeze-dev is only valid for the development split")
            current = get_existing(args.output)
            dev_ids = {q["question_id"] for q in subset}
            captured_now = {
                qid for qid, row in current.items()
                if qid in dev_ids and row.get("captured")
            }
            if len(captured_now) != 80:
                raise ValueError(
                    f"Cannot freeze development: only {len(captured_now)}/80 actual captures complete"
                )
            freeze_value = {
                "schema": 1,
                "split": "dev",
                "dev_question_count": 80,
                "captured_question_count": len(captured_now),
                "configuration_sha256": configuration["configuration_sha256"],
                "configuration": configuration,
                "frozen_utc": utc_now(),
            }
            atomic_json(args.freeze, freeze_value)
            report["dev_freeze"] = str(args.freeze)
    finally:
        if handle:
            handle.close()
            lock.unlink(missing_ok=True)

    workbook = build_workbook(
        dataset=args.dataset,
        results_path=args.output,
        output_path=args.workbook,
    )
    report["workbook"] = workbook
    report["reference_answer_leak_in_request"] = False
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
