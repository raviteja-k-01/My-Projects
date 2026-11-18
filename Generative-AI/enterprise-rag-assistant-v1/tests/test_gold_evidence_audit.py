"""Strict offline checks of gold references; never claim facts verified."""
from types import SimpleNamespace

import pytest

from scripts.audit_gold_evidence import format_question, validate_gold_references


def _data():
    chunks = {
        "SEC-B1": SimpleNamespace(
            chunk_id="SEC-B1", document_id="SEC", page_start=4,
            page_end=4, section_number="3.6", section_title="Classification",
            content_type="body", text="Record the incident reclassification time.",
        ),
        "PRIV-B1": SimpleNamespace(
            chunk_id="PRIV-B1", document_id="PRIV", page_start=7,
            page_end=7, section_number="2", section_title="Privacy",
            content_type="body", text="Escalate to Privacy.",
        ),
    }
    gold = [{
        "question_id": "Q001",
        "question": "What to record?",
        "question_type": "cross-document",
        "answerable": True,
        "expected_chunk_ids": ["SEC-B1", "PRIV-B1"],
        "expected_document_id": "SEC",
        "expected_facts": ["Reclassification requires a timestamp"],
    }]
    return gold, chunks


def test_cross_document_gold_references_are_checked_but_not_scored():
    gold, chunks = _data()
    report = validate_gold_references(gold, chunks)
    assert report["questions"] == 1
    assert report["expected_facts"] == 1
    assert report["retrieval_references"] == 2
    output = format_question(gold[0], chunks)
    assert "Record the incident reclassification time." in output
    assert "Escalate to Privacy." in output
    assert "HUMAN VALIDATION REQUIRED" in output


def test_missing_gold_chunk_refused():
    gold, chunks = _data()
    gold[0]["expected_chunk_ids"] = ["SEC-B1", "MISSING"]
    with pytest.raises(ValueError, match="Unresolved gold references"):
        validate_gold_references(gold, chunks)


def test_unmatched_primary_document_refused():
    gold, chunks = _data()
    gold[0]["expected_chunk_ids"] = ["PRIV-B1"]
    with pytest.raises(ValueError, match="wrong_primary_document"):
        validate_gold_references(gold, chunks)


def test_unanswerable_without_gold_chunk_is_allowed():
    gold, chunks = _data()
    gold[0]["expected_chunk_ids"] = []
    gold[0]["expected_document_id"] = None
    gold[0]["answerable"] = False
    report = validate_gold_references(gold, chunks)
    assert report["unanswerable"] == 1
    assert "NO SELECTED GOLD CHUNKS" in format_question(gold[0], chunks)


def test_repeated_gold_ids_refused():
    gold, chunks = _data()
    gold[0]["expected_chunk_ids"] = ["SEC-B1", "SEC-B1"]
    with pytest.raises(ValueError, match="Duplicate gold chunk IDs"):
        validate_gold_references(gold, chunks)
