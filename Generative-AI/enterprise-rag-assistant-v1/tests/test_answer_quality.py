"""All answer-evaluation tests run with zero Gemini or Qdrant requests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import evaluate_answers
from src.evaluation.answer_quality import (
    DIMENSIONS,
    attach_capture,
    audit_citation_mechanics,
    new_case,
    new_review,
    summarize_reviews,
    validate_review,
)


def gold_records():
    return [
        {
            "question_id": "Q001",
            "question": "What must employees do?",
            "question_type": "factual",
            "expected_document_id": "POLICY",
            "expected_section": "1",
            "expected_chunk_ids": ["POLICY-B001"],
            "expected_facts": ["Report the event immediately", "Notify security"],
            "answerable": True,
        },
        {
            "question_id": "Q002",
            "question": "What exact unspecified fine applies?",
            "question_type": "unanswerable",
            "expected_document_id": None,
            "expected_section": None,
            "expected_chunk_ids": [],
            "expected_facts": ["No exact fine is specified"],
            "answerable": False,
        },
    ]


def fake_response():
    evidence = {
        "evidence_id": 1,
        "chunk_id": "POLICY-B001",
        "document_id": "POLICY",
        "text": "Report the event immediately to security.",
        "page_start": 3,
    }
    return {
        "question": "What must employees do?",
        "answer": "Report the event immediately. [Evidence 1]",
        "retrieval_mode": "hybrid",
        "model": "test-offline",
        "citations": [{
            "evidence_id": 1,
            "chunk_id": "POLICY-B001",
            "document_id": "POLICY",
        }],
        "citation_count": 1,
        "evaluation_trace": {
            "retrieved_chunk_ids": ["POLICY-B001"],
            "selected_evidence": [evidence],
        },
    }


def test_mechanical_citation_validity_is_not_factual_grounding():
    response = fake_response()
    case = attach_capture(new_case(gold_records()[0]), response)
    assert case["mechanical_checks"]["mechanically_valid"] is True
    assert case["mechanical_checks"]["has_citations"] is True
    assert case["run"]["selected_evidence"][0]["text"].startswith("Report")
    # No human rating assigned by an automated citation check.
    assert all(v is None for v in new_review(gold_records()[0])["ratings"].values())

    wrong = dict(response["citations"][0], chunk_id="UNKNOWN")
    audit = audit_citation_mechanics(
        response["answer"], [wrong], response["evaluation_trace"]["selected_evidence"]
    )
    assert audit["mechanically_valid"] is False
    assert "chunk_mismatch:1" in audit["metadata_mismatches"]


def test_missing_citations_is_flagged_but_not_called_a_hallucination():
    audit = audit_citation_mechanics("The exact amount is unspecified.", [], [])
    assert audit["has_citations"] is False
    assert audit["mechanically_valid"] is True


def test_strict_capture_rejects_sparse_fallback():
    response = fake_response()
    response["retrieval_mode"] = "bm25_fallback"
    with pytest.raises(ValueError, match="Strict evaluation"):
        attach_capture(new_case(gold_records()[0]), response)


def test_prepare_does_not_overwrite_human_reviews_or_query_api(tmp_path):
    gold_file = tmp_path / "gold.jsonl"
    gold_file.write_text(
        "".join(json.dumps(row) + "\n" for row in gold_records()), encoding="utf-8"
    )
    cases_file = tmp_path / "cases.jsonl"
    review_dir = tmp_path / "reviews"

    gold, cases, reviews = evaluate_answers.prepare(gold_file, cases_file, review_dir)
    assert len(gold) == 2
    assert all(case["run"] is None for case in cases.values())
    assert all(r["ratings"]["answer_relevance"] is None for r in reviews.values())

    human_file = review_dir / "Q001.json"
    changed_review = reviews["Q001"]
    changed_review["notes"] = "Reviewed evidence details, in progress."
    human_file.write_text(json.dumps(changed_review), encoding="utf-8")

    _gold, _cases, retained = evaluate_answers.prepare(gold_file, cases_file, review_dir)
    assert retained["Q001"]["notes"] == "Reviewed evidence details, in progress."

    edited_gold = gold_records()
    edited_gold[0]["expected_facts"] = ["Different assertion"]
    gold_file.write_text(
        "".join(json.dumps(row) + "\n" for row in edited_gold), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Gold dataset changed"):
        evaluate_answers.prepare(gold_file, cases_file, review_dir)


def test_report_counts_only_completed_human_reviews():
    gold = gold_records()
    cases = {row["question_id"]: new_case(row) for row in gold}
    reviews = {row["question_id"]: new_review(row) for row in gold}
    cases["Q001"] = attach_capture(cases["Q001"], fake_response())
    summary = summarize_reviews(gold, cases, reviews)
    assert summary["captured"] == 1
    assert summary["fully_human_reviewed"] == 0
    assert summary["ratings_distribution"]["answer_relevance"] == {}

    first = reviews["Q001"]
    first["reviewer"] = "Independent reviewer"
    first["ratings"] = {dimension: 2 for dimension in DIMENSIONS}
    first["ratings"]["fact_coverage"] = 1
    first["fact_checks"][0]["status"] = "met"
    first["fact_checks"][1]["status"] = "missed"
    assert validate_review(first, gold[0]) is True
    summary = summarize_reviews(gold, cases, reviews)
    assert summary["fully_human_reviewed"] == 1
    assert summary["expected_fact_statuses"] == {"met": 1, "missed": 1}
    assert summary["ratings_distribution"]["fact_coverage"] == {1: 1}
    assert summary["ratings_distribution"]["factual_grounding"] == {2: 1}

    first["fact_checks"][0]["supporting_evidence_ids"] = [999]
    with pytest.raises(ValueError, match="unknown supporting evidence"):
        summarize_reviews(gold, cases, reviews)
    first["fact_checks"][0]["supporting_evidence_ids"] = [1]

    first["ratings"]["citation_support"] = True
    with pytest.raises(ValueError, match="Ratings must be integers"):
        validate_review(first, gold[0])


def test_repeated_live_capture_is_checkpointed_and_strict(tmp_path, monkeypatch):
    gold_file = tmp_path / "gold.jsonl"
    gold_file.write_text(
        "".join(json.dumps(row) + "\n" for row in gold_records()), encoding="utf-8"
    )
    cases_file = tmp_path / "cases.jsonl"
    review_dir = tmp_path / "reviews"
    calls = []

    class FakePipeline:
        def answer(self, question, *, evaluation_trace, allow_sparse_fallback):
            calls.append((question, evaluation_trace, allow_sparse_fallback))
            assert evaluation_trace is True
            assert allow_sparse_fallback is False
            if len(calls) == 2:
                raise RuntimeError("Simulated generation unavailable")
            return fake_response()

        def close(self):
            pass

    import src.rag.rag_pipeline as pipeline_module
    monkeypatch.setattr(pipeline_module, "EnterpriseRAGPipeline", FakePipeline)

    evaluate_answers.capture(gold_file, cases_file, review_dir, max_new=1)
    assert evaluate_answers._read_cases(cases_file)["Q001"]["run"] is not None

    # A later quota error does not discard the prior run or a human review.
    with pytest.raises(SystemExit) as exc:
        evaluate_answers.capture(gold_file, cases_file, review_dir, max_new=1)
    assert exc.value.code == 75
    saved = evaluate_answers._read_cases(cases_file)
    assert saved["Q001"]["run"] is not None
    assert saved["Q002"]["run"] is None
    assert len(calls) == 2
