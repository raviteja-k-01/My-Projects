"""Answer replay tests use no Gemini or Qdrant services."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pytest

from scripts import evaluate_answers
from src.evaluation import replay


@dataclass
class FakeChunk:
    chunk_id: str = "POLICY-B001"
    document_id: str = "POLICY"
    document_name: str = "Policy"
    page_start: int = 3
    page_end: int = 3
    section_number: str = "1"
    section_title: str = "Reporting"
    subsection_number: str | None = None
    subsection_title: str | None = None
    content_type: str = "body"
    text: str = "Report the event immediately to security."
    token_count: int = 10


def gold():
    return [{
        "question_id": "Q001",
        "question": "What must employees do?",
        "question_type": "factual",
        "expected_document_id": "POLICY",
        "expected_section": "1",
        "expected_chunk_ids": ["POLICY-B001"],
        "expected_facts": ["Report the event immediately"],
        "answerable": True,
    }]


def saved_checkpoint():
    return {"Q001": {"retrieved_chunk_ids": ["POLICY-B001"]}}


def test_replay_loads_verified_corpus_and_prior_rankings(monkeypatch, tmp_path):
    monkeypatch.setattr(replay, "verify_corpus", lambda path: [FakeChunk()])
    monkeypatch.setattr(
        replay, "load_checkpoint", lambda checkpoint_path, gold_records: saved_checkpoint()
    )
    checkpoints, by_chunk = replay.load_replay_inputs(
        gold=gold(), checkpoint_path=tmp_path / "hybrid.jsonl", chunks_path=tmp_path / "chunks.jsonl"
    )
    candidates = replay.replay_candidates("Q001", checkpoints, by_chunk)
    assert len(candidates) == 1
    assert candidates[0]["rank"] == 1
    assert candidates[0]["chunk"]["text"] == FakeChunk().text
    assert candidates[0]["chunk_id"] == "POLICY-B001"


def test_replay_refuses_unknown_ranking_ids(monkeypatch, tmp_path):
    monkeypatch.setattr(replay, "verify_corpus", lambda path: [FakeChunk()])
    monkeypatch.setattr(
        replay, "load_checkpoint",
        lambda *args: {"Q001": {"retrieved_chunk_ids": ["NOT_IN_CORPUS"]}},
    )
    with pytest.raises(ValueError, match="Unknown retrieved chunk ID"):
        replay.load_replay_inputs(
            gold=gold(), checkpoint_path=tmp_path / "checkpoint",
            chunks_path=tmp_path / "chunks",
        )


def test_generation_only_capture_does_not_initialize_live_retrieval(
    monkeypatch, tmp_path, capsys
):
    gold_file = tmp_path / "gold.jsonl"
    gold_file.write_text(json.dumps(gold()[0]) + "\n", encoding="utf-8")
    cases_file = tmp_path / "cases.jsonl"
    review_dir = tmp_path / "reviews"
    checks = {"generated": 0}

    def get_saved(*, gold, checkpoint_path, chunks_path):
        assert len(gold) == 1
        return saved_checkpoint(), {"POLICY-B001": vars(FakeChunk())}

    monkeypatch.setattr(replay, "load_replay_inputs", get_saved)

    from src.generation import answer_generator
    from src.retrieval import hybrid_retriever
    from src.indexing import embedding_client

    class FakeGenerator:
        def generate(self, *, question, context):
            checks["generated"] += 1
            assert question == "What must employees do?"
            assert "Report the event immediately" in context
            return {
                "answer": "Report the event immediately. [Evidence 1]",
                "model": "fake-generation",
                "usage": {},
            }

    monkeypatch.setattr(answer_generator, "GroundedAnswerGenerator", FakeGenerator)
    monkeypatch.setattr(hybrid_retriever, "HybridRetriever",
                        lambda: pytest.fail("New retrieval must never be initialized"))
    monkeypatch.setattr(embedding_client, "GeminiEmbeddingClient",
                        lambda: pytest.fail("Embedding must never be initialized"))

    evaluate_answers.capture_replay(
        gold_path=gold_file, cases_path=cases_file, review_dir=review_dir,
        chunks_path=tmp_path / "chunks", checkpoint_path=tmp_path / "checkpoint",
        max_new=1,
    )
    cases = evaluate_answers._read_cases(cases_file)
    run = cases["Q001"]["run"]
    assert run["retrieval_mode"] == "hybrid"
    assert run["retrieval_source"] == "saved_hybrid_checkpoint"
    assert run["latency"]["retrieval_seconds"] is None
    assert run["selected_evidence"][0]["text"] == FakeChunk().text
    assert cases["Q001"]["mechanical_checks"]["mechanically_valid"] is True
    assert checks["generated"] == 1
    assert (review_dir / "Q001.json").exists()

    # A second run reuses the answer rather than calling the model.
    evaluate_answers.capture_replay(
        gold_path=gold_file, cases_path=cases_file, review_dir=review_dir,
        chunks_path=tmp_path / "chunks", checkpoint_path=tmp_path / "checkpoint",
        max_new=1,
    )
    assert checks["generated"] == 1
    assert "NO PENDING SAVED HYBRID RESULTS" in capsys.readouterr().out


def test_preview_is_offline_and_shows_provenance(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        replay, "load_replay_inputs",
        lambda **kwargs: (saved_checkpoint(), {"POLICY-B001": vars(FakeChunk())}),
    )
    monkeypatch.setattr(evaluate_answers, "load_gold_questions", lambda path: gold())
    evaluate_answers.preview_replay(
        question_id="Q001",
        checkpoint_path=tmp_path / "checkpoint",
        chunks_path=tmp_path / "chunks",
    )
    output = capsys.readouterr().out
    assert "OFFLINE HYBRID REPLAY" in output
    assert "POLICY-B001" in output
    assert "saved_hybrid_checkpoint" in output


def test_generation_only_capture_refuses_missing_checkpoint(monkeypatch, tmp_path):
    gold_file = tmp_path / "gold.jsonl"
    gold_file.write_text(json.dumps(gold()[0]) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        replay, "load_replay_inputs",
        lambda **kwargs: ({}, {"POLICY-B001": vars(FakeChunk())}),
    )
    evaluate_answers.capture_replay(
        gold_path=gold_file,
        cases_path=tmp_path / "cases.jsonl",
        review_dir=tmp_path / "reviews",
        max_new=1,
    )
    cases = evaluate_answers._read_cases(tmp_path / "cases.jsonl")
    assert cases["Q001"]["run"] is None


def test_confirmed_empty_retrieval_captures_abstention_without_model(monkeypatch, tmp_path):
    from src.generation import answer_generator
    from scripts.evaluate_hybrid_retrieval_resumable import _build_result
    from scripts.build_dense_index import EXPECTED_CHUNKS_SHA256
    row = dict(gold()[0], answerable=False, expected_chunk_ids=[])
    gold_file = tmp_path / "gold.jsonl"
    gold_file.write_text(json.dumps(row) + "\n")
    checkpoint = _build_result(row, [], provenance={
        "schema_version": 1, "retrieval_mode": "hybrid", "retrieval_performed": True,
        "corpus_sha256": EXPECTED_CHUNKS_SHA256,
    })
    monkeypatch.setattr(replay, "load_replay_inputs", lambda **kwargs: ({"Q001": checkpoint}, {}))
    monkeypatch.setattr(answer_generator, "GroundedAnswerGenerator",
                        lambda: pytest.fail("Empty evidence needs no model or credentials"))
    evaluate_answers.capture_replay(gold_path=gold_file, cases_path=tmp_path / "cases",
                                    review_dir=tmp_path / "reviews")
    run = evaluate_answers._read_cases(tmp_path / "cases")["Q001"]["run"]
    assert "insufficient evidence" in run["answer"]
    assert run["model"] is None
    assert run["citations"] == []


def test_legacy_skipped_retrieval_is_not_treated_as_abstention_evidence():
    checkpoint = {"retrieved_chunk_ids": [], "expected_chunk_ids": []}
    assert replay.is_replayable(checkpoint) is False
    with pytest.raises(ValueError, match="Legacy skipped"):
        replay.replay_candidates("Q027", {"Q027": checkpoint}, {})


@pytest.mark.parametrize("bad_record, expected", [
    ({"retrieved_chunk_ids": ["POLICY-B001", "POLICY-B001"]}, "Duplicate retrieved"),
    ({"retrieved_chunk_ids": ["POLICY-B001"], "provenance": {"corpus_sha256": "0" * 64}}, "SHA-256 mismatch"),
])
def test_replay_refuses_duplicate_ranks_and_mismatched_provenance(monkeypatch, tmp_path, bad_record, expected):
    monkeypatch.setattr(replay, "verify_corpus", lambda path: [FakeChunk()])
    monkeypatch.setattr(replay, "load_checkpoint", lambda *args: {"Q001": bad_record})
    with pytest.raises(ValueError, match=expected):
        replay.load_replay_inputs(gold=gold(), checkpoint_path=tmp_path / "checkpoint",
                                 chunks_path=tmp_path / "chunks")


def test_replay_refuses_duplicate_corpus_ids(monkeypatch, tmp_path):
    monkeypatch.setattr(replay, "verify_corpus", lambda path: [FakeChunk(), FakeChunk()])
    with pytest.raises(ValueError, match="Duplicate chunk IDs"):
        replay.load_replay_inputs(gold=gold(), checkpoint_path=tmp_path / "checkpoint",
                                 chunks_path=tmp_path / "chunks")
