"""Offline regressions for resumable live-evaluation automation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_live_evaluation as runner


def test_live_preview_creates_workbook_without_api(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(runner.requests, "post", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("Preview must not call the API")
    ))
    output = tmp_path / "live.jsonl"
    workbook = tmp_path / "report.xlsx"
    result = runner.main([
        "--output", str(output),
        "--workbook", str(workbook),
        "--max-new", "0",
    ])
    assert result == 0
    assert workbook.exists()
    assert not output.exists()


def test_live_capture_is_resumable_and_does_not_send_reference_answers(monkeypatch, tmp_path: Path):
    payloads = []

    class Response:
        status_code = 200

        def json(self):
            return {
                "answer": "Only PUBLIC information [Evidence 1]",
                "citations": [{
                    "evidence_id": 1,
                    "chunk_id": "AI-GOV-05-B0035",
                    "document_id": "AI-GOV-05",
                }],
                "evidence_count": 1,
                "retrieval_mode": "hybrid",
                "retrieval_source": "live",
                "model": "test-model",
                "latency": {"total_seconds": 0.25},
                "evaluation_trace": {
                    "retrieved_chunk_ids": ["AI-GOV-05-B0035"],
                    "selected_evidence": [{
                        "evidence_id": 1,
                        "chunk_id": "AI-GOV-05-B0035",
                        "document_id": "AI-GOV-05",
                        "document_name": "Test",
                        "text": "Only PUBLIC information",
                        "page_start": 24,
                        "page_end": 24,
                        "content_type": "body",
                    }],
                },
            }

    def post(url, json, headers, timeout):
        payloads.append(json)
        return Response()

    monkeypatch.setattr(runner.requests, "post", post)
    output = tmp_path / "live.jsonl"
    workbook = tmp_path / "report.xlsx"
    runner.main([
        "--output", str(output),
        "--workbook", str(workbook),
        "--max-new", "1",
        "--confirm-live",
        "--request-budget", "1",
    ])
    assert len(payloads) == 1
    assert "reference_answer" not in payloads[0]
    assert payloads[0]["evaluation_trace"] is True
    rows = runner.read_results(output)
    assert len(rows) == 1
    assert rows[0]["question_id"] == "E001"
    assert rows[0]["captured"] is True
    assert workbook.exists()

    runner.main([
        "--output", str(output),
        "--workbook", str(workbook),
        "--max-new", "0",
    ])
    assert len(payloads) == 1


def test_test_split_requires_explicit_unlock(tmp_path: Path):
    with pytest.raises(ValueError, match="Held-out test"):
        runner.main([
            "--split", "test",
            "--output", str(tmp_path / "live.jsonl"),
            "--workbook", str(tmp_path / "report.xlsx"),
            "--max-new", "0",
        ])


def test_persistent_budget_cannot_change_silently(monkeypatch, tmp_path: Path):
    class Response:
        status_code = 200

        def json(self):
            return {
                "answer": "Only PUBLIC information [Evidence 1]",
                "citations": [],
                "evidence_count": 1,
                "retrieval_mode": "hybrid",
                "retrieval_source": "live",
                "model": "test-model",
                "latency": {"total_seconds": 0.1},
                "evaluation_trace": {
                    "retrieved_chunk_ids": ["AI-GOV-05-B0035"],
                    "selected_evidence": [],
                },
            }

    monkeypatch.setattr(runner.requests, "post", lambda *args, **kwargs: Response())
    output = tmp_path / "live.jsonl"
    workbook = tmp_path / "report.xlsx"
    runner.main([
        "--output", str(output),
        "--workbook", str(workbook),
        "--max-new", "1",
        "--confirm-live",
        "--request-budget", "1",
    ])
    with pytest.raises(ValueError, match="Persistent request budget"):
        runner.main([
            "--output", str(output),
            "--workbook", str(workbook),
            "--max-new", "1",
            "--confirm-live",
            "--request-budget", "2",
        ])


def test_development_freeze_requires_all_80(tmp_path: Path):
    with pytest.raises(ValueError, match="Cannot freeze development"):
        runner.main([
            "--output", str(tmp_path / "empty.jsonl"),
            "--workbook", str(tmp_path / "report.xlsx"),
            "--max-new", "0",
            "--freeze-dev",
        ])
