"""Offline retrieval-review packet contract: source locks and no automatic labels."""
from __future__ import annotations

import json

import pytest

from src.evaluation.benchmark120_evidence import sha256, read_jsonl
from src.evaluation.benchmark120_hybrid_capture import _question_digest
from src.evaluation.benchmark120_retrieval_review import export_review_packet


def make_sources(tmp_path):
    dataset = tmp_path / "questions.jsonl"
    corpus = tmp_path / "chunks.jsonl"
    retrieval = tmp_path / "retrieval.jsonl"
    questions = [
        {
            "question_id": "E001", "split": "dev", "answerable": True,
            "question": "Which control is required?",
            "expected_facts": ["An approval is required"],
            "expected_documents": ["POL-1"], "evidence_documents": ["POL-1"],
            "evidence": [["An approval is required"]],
        },
        {
            "question_id": "E002", "split": "dev", "answerable": False,
            "question": "What is the exact secret amount?",
            "expected_facts": ["Not specified"],
            "expected_documents": ["POL-1"], "evidence_documents": ["POL-1"],
            "evidence": [],
        },
        {
            "question_id": "E003", "split": "test", "answerable": True,
            "question": "Held-out secret question",
            "expected_facts": ["Heldout"],
            "expected_documents": ["POL-2"], "evidence_documents": ["POL-2"],
            "evidence": [],
        },
    ]
    chunks = [
        {"chunk_id": "A", "document_id": "POL-1", "page_start": 1,
         "page_end": 1, "text": "Background rule with no approval wording."},
        {"chunk_id": "B", "document_id": "POL-1", "page_start": 2,
         "page_end": 2, "text": "An approval is required before privileged access."},
    ]
    dataset.write_text("".join(json.dumps(x) + "\n" for x in questions), encoding="utf-8")
    corpus.write_text("".join(json.dumps(x) + "\n" for x in chunks), encoding="utf-8")
    retrieval.write_text("".join(json.dumps({
        "question_id": q["question_id"], "split": q["split"],
        "question_sha256": _question_digest(q["question"]),
        "source_dataset_sha256": sha256(dataset),
        "source_corpus_sha256": sha256(corpus),
        "retrieval_mode": "strict_hybrid",
        "retrieval_limits": {"dense_limit": 20, "sparse_limit": 20, "fused_limit": 20},
        "retrieved_chunk_ids": ["A", "B"],
    }) + "\n" for q in questions if q["split"] == "dev"), encoding="utf-8")
    return dict(dataset_path=dataset, corpus_path=corpus, retrieval_path=retrieval)


def test_packets_show_actual_rankings_and_independent_anchor_hints(tmp_path):
    sources = make_sources(tmp_path)
    out = tmp_path / "packet"
    summary = export_review_packet(**sources, output_dir=out)
    assert summary["questions_exported"] == 1
    assert summary["question_ids"] == ["E001"]
    assert summary["verified_labels"] == 0
    assert all(summary[name] == 0 for name in (
        "new_embedding_calls", "new_qdrant_queries", "new_generation_calls"
    ))

    rows = read_jsonl(out / "retrieval_review_packet.jsonl")
    assert len(rows) == 1
    row = rows[0]
    assert row["ranked_top_k"][0]["chunk_id"] == "A"
    assert row["ranked_top_k"][1]["chunk_id"] == "B"
    assert row["source_anchor_matches_unverified"][0]["chunk_ids"] == ["B"]
    assert row["source_anchor_excerpts_unverified"][0]["text"] == (
        "An approval is required before privileged access."
    )
    assert "Held-out secret question" not in str(row)
    assert row["source_corpus_sha256"] == sha256(sources["corpus_path"])

    labels = read_jsonl(out / "relevance_labels_UNREVIEWED.jsonl")
    assert len(labels) == 1
    assert labels[0]["verified_relevant_chunk_ids"] == []
    assert labels[0]["review_state"] == "unreviewed"
    assert labels[0]["reviewer"] is None
    assert labels[0]["candidate_chunk_ids_unverified"] == ["A", "B"]


def test_reject_overwrite_and_bad_ranges(tmp_path):
    sources = make_sources(tmp_path)
    out = tmp_path / "packet"
    with pytest.raises(ValueError, match="start"):
        export_review_packet(**sources, output_dir=out, start=-1)
    with pytest.raises(ValueError, match="No answerable"):
        export_review_packet(**sources, output_dir=out, start=1)
    assert not out.exists()
    export_review_packet(**sources, output_dir=out)
    with pytest.raises(ValueError, match="exists"):
        export_review_packet(**sources, output_dir=out)


def test_tampered_source_refused_without_writing_packet(tmp_path):
    sources = make_sources(tmp_path)
    sources["corpus_path"].write_text(
        sources["corpus_path"].read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    out = tmp_path / "packet"
    with pytest.raises(ValueError, match="checkpoint"):
        export_review_packet(**sources, output_dir=out)
    assert not out.exists()


def test_top_k_limit_cannot_be_unbounded(tmp_path):
    sources = make_sources(tmp_path)
    with pytest.raises(ValueError, match="top_k"):
        export_review_packet(**sources, output_dir=tmp_path / "packet", top_k=999)
