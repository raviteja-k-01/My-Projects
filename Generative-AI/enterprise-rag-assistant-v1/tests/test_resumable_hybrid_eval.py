from __future__ import annotations

import pytest

from scripts import evaluate_hybrid_retrieval_resumable as evaluator


def _gold():
    return [
        {"question_id": "Q1", "question": "first question", "question_type": "factual", "expected_chunk_ids": ["A"]},
        {"question_id": "Q2", "question": "second question", "question_type": "factual", "expected_chunk_ids": ["B"]},
        {"question_id": "Q3", "question": "unanswerable", "question_type": "unanswerable", "expected_chunk_ids": []},
    ]


def test_checkpoint_roundtrip_and_refuses_changed_gold(tmp_path):
    gold = _gold()
    output = tmp_path / "evaluation.jsonl"
    first = evaluator._build_result(gold[0], ["A", "C"])
    evaluator.save_checkpoint(output, {"Q1": first}, gold)
    assert evaluator.load_checkpoint(output, gold) == {"Q1": first}
    changed = _gold()
    changed[0]["expected_chunk_ids"] = ["Z"]
    with pytest.raises(ValueError, match="Stale/inconsistent"):
        evaluator.load_checkpoint(output, changed)


def test_resume_skips_completed_queries_after_failure(tmp_path, monkeypatch):
    gold = _gold()
    monkeypatch.setattr(evaluator, "load_gold_questions", lambda: gold)
    called = []

    class FakeHybridRetriever:
        def retrieve(self, query, **kwargs):
            called.append(query)
            if query == "second question" and called.count("second question") == 1:
                raise RuntimeError("Simulated daily quota exhaustion")
            return [{"chunk_id": "A" if query == "first question" else "B"}]

        def close(self):
            return None

    monkeypatch.setattr(evaluator, "HybridRetriever", FakeHybridRetriever)
    output = tmp_path / "evaluation.jsonl"

    with pytest.raises(RuntimeError, match="daily quota"):
        evaluator.run(output=output)
    first_pass = evaluator.load_checkpoint(output, gold)
    assert list(first_pass) == ["Q1"]
    assert called == ["first question", "second question"]

    evaluator.run(output=output)
    second_pass = evaluator.load_checkpoint(output, gold)
    assert list(second_pass) == ["Q1", "Q2", "Q3"]
    assert called == ["first question", "second question", "second question"]
    assert second_pass["Q1"]["metrics"]["hit_at_k"]["hit@5"] == 1.0
    assert second_pass["Q3"]["metrics"]["evaluable"] is False


def test_max_new_produces_usable_checkpoint(tmp_path, monkeypatch):
    gold = _gold()
    monkeypatch.setattr(evaluator, "load_gold_questions", lambda: gold)

    class FakeHybridRetriever:
        def retrieve(self, query, **kwargs):
            return [{"chunk_id": "A"}]

        def close(self):
            pass

    monkeypatch.setattr(evaluator, "HybridRetriever", FakeHybridRetriever)
    output = tmp_path / "evaluation.jsonl"
    evaluator.run(output=output, max_new=1)
    assert list(evaluator.load_checkpoint(output, gold)) == ["Q1"]
