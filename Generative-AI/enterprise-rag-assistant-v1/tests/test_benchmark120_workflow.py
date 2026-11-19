"""No network, model, embedding, or fake benchmark labels."""
import json
from pathlib import Path

import pytest

from src.evaluation.benchmark120_evidence import audit
from scripts.evaluate_benchmark120 import (
    initialize_relevance_labels, load_dataset, score_retrieval, summary,
)


def sample(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    corpus = tmp_path / "chunks.jsonl"
    rows = [
        {"question_id": "X1", "split": "dev", "answerable": True,
         "question": "Who approves?", "expected_facts": ["The approver is Jane."],
         "expected_documents": ["DOC-A"], "evidence_documents": ["DOC-A"],
         "evidence_locations": [{"document_id": "DOC-A", "page": 1}],
         "evidence": [["Approver Jane"]]},
        {"question_id": "X2", "split": "test", "answerable": True,
         "question": "Which team?", "expected_facts": ["The review team."],
         "expected_documents": ["DOC-B"], "evidence_documents": ["DOC-B"],
         "evidence_locations": [{"document_id": "DOC-B", "page": 2}],
         "evidence": [["Review team"]]},
        {"question_id": "X3", "split": "dev", "answerable": False,
         "question": "What exact amount?", "expected_facts": ["No exact amount given."],
         "expected_documents": ["DOC-A"], "evidence_documents": [],
         "evidence_locations": [], "evidence": []},
    ]
    chunks = [
        {"chunk_id": "C1", "document_id": "DOC-A", "page_start": 1, "page_end": 1,
         "content_type": "table", "text": "Row 1: Field: Approver | Controlled value: Jane"},
        {"chunk_id": "C2", "document_id": "DOC-B", "page_start": 2, "page_end": 2,
         "text": "Review team handles the request."},
    ]
    dataset.write_text("".join(json.dumps(q) + "\n" for q in rows))
    corpus.write_text("".join(json.dumps(c) + "\n" for c in chunks))
    return dataset, corpus


def test_status_counts_and_splits(tmp_path):
    dataset, _ = sample(tmp_path)
    report = summary(dataset)
    assert report["by_split"] == {"dev": 2, "test": 1}
    assert report["answerable_count"] == 2
    assert report["unanswerable_count"] == 1
    assert report["expected_fact_prompts"] == 3


def test_bad_duplicate_questions_rejected(tmp_path):
    dataset, _ = sample(tmp_path)
    with dataset.open("a") as f:
        f.write(dataset.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="repeated"):
        load_dataset(dataset)


def test_labels_not_assigned_from_evidence_matches(tmp_path):
    dataset, corpus = sample(tmp_path)
    audit(dataset, corpus, tmp_path / "audit")
    path = tmp_path / "labels.jsonl"
    info = initialize_relevance_labels(dataset, tmp_path / "audit", path)
    assert info == {"records": 3, "unreviewed": 3}
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    assert rows[0]["candidate_chunk_ids_unverified"] == ["C1"]
    assert all(q["review_state"] == "unreviewed" and q["reviewer"] is None
               and q["verified_relevant_chunk_ids"] == [] for q in rows)
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        initialize_relevance_labels(dataset, tmp_path / "audit", path)


def test_stale_audit_is_rejected(tmp_path):
    dataset, corpus = sample(tmp_path)
    audit(dataset, corpus, tmp_path / "audit")
    dataset.write_text(dataset.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        initialize_relevance_labels(dataset, tmp_path / "audit", tmp_path / "labels.jsonl")


def test_no_metric_when_human_labels_unverified(tmp_path):
    dataset, corpus = sample(tmp_path)
    audit(dataset, corpus, tmp_path / "audit")
    labels = tmp_path / "labels.jsonl"
    initialize_relevance_labels(dataset, tmp_path / "audit", labels)
    retrievals = tmp_path / "retrieval.jsonl"
    retrievals.write_text(json.dumps({"question_id": "X1", "retrieved_chunk_ids": ["C1"]}) + "\n")
    result = score_retrieval(dataset, labels, retrievals, "dev", [1, 5])
    assert result["scored_answerable_questions"] == 0
    assert result["metrics"] == {}
    assert result["skipped"]["not_human_verified"] == 1
    assert result["skipped"]["unanswerable_not_scored_as_retrieval_failure"] == 1


def test_explicitly_reviewed_labels_can_score(tmp_path):
    dataset, corpus = sample(tmp_path)
    audit(dataset, corpus, tmp_path / "audit")
    labels = tmp_path / "labels.jsonl"
    initialize_relevance_labels(dataset, tmp_path / "audit", labels)
    data = [json.loads(line) for line in labels.read_text().splitlines()]
    data[0].update(review_state="verified", reviewer="Human reviewer", verified_relevant_chunk_ids=["C1"])
    labels.write_text("".join(json.dumps(row) + "\n" for row in data))
    checkpoint = tmp_path / "retrieval.jsonl"
    checkpoint.write_text(json.dumps({"question_id": "X1", "retrieved_chunk_ids": ["C2", "C1"]}) + "\n")
    result = score_retrieval(dataset, labels, checkpoint, "dev", [1, 2])
    assert result["scored_answerable_questions"] == 1
    assert result["metrics"]["recall@1"] == 0.0
    assert result["metrics"]["recall@2"] == 1.0
    assert result["metrics"]["mrr@2"] == 0.5


def test_test_split_requires_its_own_metrics(tmp_path):
    dataset, corpus = sample(tmp_path)
    audit(dataset, corpus, tmp_path / "audit")
    labels = tmp_path / "labels.jsonl"
    initialize_relevance_labels(dataset, tmp_path / "audit", labels)
    data = [json.loads(line) for line in labels.read_text().splitlines()]
    data[1].update(review_state="verified", reviewer="Human reviewer", verified_relevant_chunk_ids=["C2"])
    labels.write_text("".join(json.dumps(row) + "\n" for row in data))
    checkpoint = tmp_path / "retrieval.jsonl"
    checkpoint.write_text(json.dumps({"question_id": "X2", "retrieved_chunk_ids": ["C2"]}) + "\n")
    assert score_retrieval(dataset, labels, checkpoint, "dev", [1])["scored_answerable_questions"] == 0
    test_result = score_retrieval(dataset, labels, checkpoint, "test", [1])
    assert test_result["scored_answerable_questions"] == 1
    assert test_result["metrics"]["recall@1"] == 1.0


def test_verified_label_without_reviewer_rejected(tmp_path):
    dataset, corpus = sample(tmp_path)
    audit(dataset, corpus, tmp_path / "audit")
    labels = tmp_path / "labels.jsonl"
    initialize_relevance_labels(dataset, tmp_path / "audit", labels)
    data = [json.loads(line) for line in labels.read_text().splitlines()]
    data[0]["review_state"] = "verified"
    labels.write_text("".join(json.dumps(row) + "\n" for row in data))
    checkpoint = tmp_path / "checkpoint.jsonl"
    checkpoint.write_text(json.dumps({"question_id": "X1", "retrieved_chunk_ids": ["C1"]}) + "\n")
    with pytest.raises(ValueError, match="require reviewer"):
        score_retrieval(dataset, labels, checkpoint, "dev", [1])
