from __future__ import annotations

import json
import pytest

from scripts import evaluate_hybrid_retrieval_resumable as evaluator


@pytest.fixture(autouse=True)
def offline_corpus(monkeypatch):
    from src.evaluation import replay
    from scripts.build_dense_index import EXPECTED_CHUNKS_SHA256
    monkeypatch.setattr(replay, "load_replay_inputs", lambda **kwargs: (
        evaluator.load_checkpoint(kwargs["checkpoint_path"], kwargs["gold"]),
        {key: {"chunk_id": key} for key in ("A", "B", "C")},
    ))
    monkeypatch.setattr(evaluator, "retrieval_provenance", lambda path: {
        "schema_version": 1, "retrieval_mode": "hybrid", "retrieval_performed": True,
        "corpus_sha256": EXPECTED_CHUNKS_SHA256,
    })


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
    monkeypatch.setattr(evaluator, "load_gold_questions", lambda path: gold)
    called = []

    class FakeHybridRetriever:
        def retrieve(self, query, **kwargs):
            called.append(query)
            if query == "second question" and called.count("second question") == 1:
                raise RuntimeError("Simulated daily quota exhaustion")
            key = "A" if query == "first question" else "B"
            return [{"chunk_id": key, "chunk": {"chunk_id": key}}]

        def close(self):
            return None

    monkeypatch.setattr(evaluator, "HybridRetriever", FakeHybridRetriever)
    output = tmp_path / "evaluation.jsonl"

    with pytest.raises(RuntimeError, match="daily quota"):
        evaluator.run(output=output, max_new=3)
    first_pass = evaluator.load_checkpoint(output, gold)
    assert list(first_pass) == ["Q1"]
    assert called == ["first question", "second question"]

    evaluator.run(output=output, max_new=3)
    second_pass = evaluator.load_checkpoint(output, gold)
    assert list(second_pass) == ["Q1", "Q2", "Q3"]
    assert called == ["first question", "second question", "second question", "unanswerable"]
    assert second_pass["Q1"]["metrics"]["hit_at_k"]["hit@5"] == 1.0
    assert second_pass["Q3"]["metrics"]["evaluable"] is False
    assert second_pass["Q3"]["retrieved_chunk_ids"] == ["B"]


def test_max_new_produces_usable_checkpoint(tmp_path, monkeypatch):
    gold = _gold()
    monkeypatch.setattr(evaluator, "load_gold_questions", lambda path: gold)

    class FakeHybridRetriever:
        def retrieve(self, query, **kwargs):
            return [{"chunk_id": "A", "chunk": {"chunk_id": "A"}}]

        def close(self):
            pass

    monkeypatch.setattr(evaluator, "HybridRetriever", FakeHybridRetriever)
    output = tmp_path / "evaluation.jsonl"
    evaluator.run(output=output, max_new=1)
    assert list(evaluator.load_checkpoint(output, gold)) == ["Q1"]


def test_duplicate_ranked_ids_are_rejected(tmp_path):
    record = evaluator._build_result(_gold()[0], ["A", "C"])
    record["retrieved_chunk_ids"] = ["A", "A"]
    output = tmp_path / "checkpoint.jsonl"
    output.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="Duplicate retrieved"):
        evaluator.load_checkpoint(output, _gold())


def test_corrupt_line_is_reported_and_original_preserved(tmp_path):
    output = tmp_path / "checkpoint.jsonl"
    output.write_text('{"question_id":', encoding="utf-8")
    before = output.read_bytes()
    with pytest.raises(ValueError, match="Checkpoint line 1"):
        evaluator.load_checkpoint(output, _gold())
    assert output.read_bytes() == before


def test_atomic_checkpoint_interruption_preserves_previous_file(tmp_path, monkeypatch):
    from pathlib import Path
    output = tmp_path / "checkpoint.jsonl"
    first = evaluator._build_result(_gold()[0], ["A"])
    evaluator.save_checkpoint(output, {"Q1": first}, _gold())
    before = output.read_bytes()
    real_replace = Path.replace
    monkeypatch.setattr(Path, "replace", lambda *args: (_ for _ in ()).throw(OSError("interrupted")))
    with pytest.raises(OSError, match="interrupted"):
        evaluator.save_checkpoint(output, {"Q1": first, "Q2": evaluator._build_result(_gold()[1], ["B"])}, _gold())
    assert output.read_bytes() == before
    monkeypatch.setattr(Path, "replace", real_replace)
    assert evaluator.load_checkpoint(output, _gold()) == {"Q1": first}


def test_processing_complete_does_not_claim_quality_pass(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(evaluator, "load_gold_questions", lambda path: _gold())
    monkeypatch.setattr(evaluator, "HybridRetriever", lambda: type("Retriever", (), {
        "retrieve": lambda self, **kwargs: [{"chunk_id": "C", "chunk": {"chunk_id": "C"}}],
        "close": lambda self: None,
    })())
    evaluator.run(output=tmp_path / "checkpoint", max_new=3)
    output = capsys.readouterr().out
    assert "MRR: 0.0000" in output
    assert "PROCESSING: COMPLETE" in output
    assert "PASS" not in output
