from copy import deepcopy
from pathlib import Path
import pytest

from src.evaluation.answer_quality import new_case, new_review
from src.evaluation.progress import build_progress


def test_progress_does_not_confuse_missing_gold_with_unanswerability():
    gold = [{"question_id": "Q1", "question": "Question", "question_type": "unanswerable",
             "answerable": False, "expected_chunk_ids": [], "expected_facts": ["Abstain"]}]
    cases = {"Q1": new_case(gold[0])}
    reviews = {"Q1": new_review(gold[0])}
    before = deepcopy(reviews)
    row = build_progress(gold, cases, reviews, {}, {})["questions"][0]
    assert row["status"] == "NOT STARTED"
    assert row["mechanical_citations_valid"] is None
    blocked = build_progress(gold, cases, reviews, {}, {}, retrieval_blocker="Observed embedding HTTP 429")
    assert blocked["questions"][0]["status"] == "BLOCKED"
    assert blocked["questions"][0]["retrieval_checkpoint"] == "missing"
    checkpoints = {"Q1": {"retrieved_chunk_ids": [], "expected_chunk_ids": []}}
    assert build_progress(gold, cases, reviews, checkpoints, {})["questions"][0]["status"] == "BLOCKED"
    checkpoints["Q1"]["provenance"] = {"retrieval_performed": True}
    assert build_progress(gold, cases, reviews, checkpoints, {})["questions"][0]["status"] == "RETRIEVAL COMPLETE"
    assert reviews == before


@pytest.mark.parametrize("destination", [
    "data/processed/answer_eval_cases.jsonl",
    "data/processed/hybrid_retrieval_live.jsonl",
    "data/evaluation/gold_questions.jsonl",
    "data/processed/answer_review_forms/Q001.json",
    "corpus.jsonl",
])
def test_status_rejects_destructive_report_destination_before_reading(monkeypatch, destination):
    from scripts import evaluate_answers
    monkeypatch.setattr(evaluate_answers, "prepare", lambda: pytest.fail("Must reject before any writes"))
    with pytest.raises(ValueError, match="protected"):
        evaluate_answers.status(chunks_path=Path("corpus.jsonl"), output=Path(destination))
