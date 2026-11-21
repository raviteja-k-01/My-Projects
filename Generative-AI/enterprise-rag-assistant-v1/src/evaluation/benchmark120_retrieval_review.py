"""Offline human-review packets for source-locked RAG-120 development retrieval.

No model calls, no external services and NO automatically verified labels.
The file preserves complete top-K texts and independent source-anchor hints
to help a reviewer determine relevance against the question and the corpus.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from src.evaluation.benchmark120_answer_replay import load_inputs
from src.evaluation.benchmark120_evidence import Chunk, assess_snippet, snippet_list


def export_review_packet(
    *, dataset_path: Path, corpus_path: Path, retrieval_path: Path,
    output_dir: Path, start: int = 0, limit: int = 5, top_k: int = 10,
) -> dict[str, Any]:
    """Export unreviewed dev labels + ranked source excerpts for manual review.

    Source candidate hints are found independently of retrieval ranking but
    are NOT necessarily complete, relevant, or proof of factual entailment.
    Recall denominators require independent review of ALL relevant evidence.
    """
    if any(type(v) is not int for v in (start, limit, top_k)):
        raise ValueError("start, limit and top_k must be integers")
    if start < 0 or limit < 1 or not 1 <= top_k <= 20:
        raise ValueError("Require start >= 0, limit >= 1 and 1 <= top_k <= 20")

    questions, chunks, retrieval, digests = load_inputs(
        dataset_path=dataset_path, corpus_path=corpus_path,
        retrieval_path=retrieval_path, split="dev", unlock_test=False,
    )
    eligible = [q for q in questions if q["answerable"] is True and q["question_id"] in retrieval]
    selected = eligible[start:start + limit]
    if not selected:
        raise ValueError("No answerable, checkpointed development questions in that range")
    # Prepared only after verification; prevents silent overwriting of review work.
    if output_dir.exists():
        raise ValueError("Review output directory already exists; refusing overwrite")

    by_document: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks.values():
        record = Chunk.from_row(chunk)
        by_document[record.document_id].append(record)

    review_rows = []
    label_rows = []
    for q in selected:
        qid = q["question_id"]
        ranked_ids = retrieval[qid]["retrieved_chunk_ids"]
        anchors = [
            assess_snippet(q, snippet, by_document)
            for snippet in snippet_list(q.get("evidence", []))
        ]
        anchor_ids = list(dict.fromkeys(
            cid for anchor in anchors for cid in anchor["chunk_ids"]
        ))
        ranked = [{
            "rank": index,
            "chunk_id": cid,
            "document_id": chunks[cid].get("document_id"),
            "page_start": chunks[cid].get("page_start"),
            "page_end": chunks[cid].get("page_end"),
            "text": chunks[cid]["text"],
        } for index, cid in enumerate(ranked_ids[:top_k], 1)]
        anchor_excerpts = [{
            "chunk_id": cid,
            "document_id": chunks[cid].get("document_id"),
            "page_start": chunks[cid].get("page_start"),
            "page_end": chunks[cid].get("page_end"),
            "text": chunks[cid]["text"],
            "present_in_top_k": cid in ranked_ids[:top_k],
        } for cid in anchor_ids]
        review_rows.append({
            "question_id": qid,
            "split": "dev",
            "question": q["question"],
            "expected_documents_unverified": q["expected_documents"],
            "expected_facts_unverified": q["expected_facts"],
            "source_dataset_sha256": digests["dataset"],
            "source_corpus_sha256": digests["corpus"],
            "source_retrieval_sha256": digests["retrieval"],
            "ranked_top_k": ranked,
            "source_anchor_matches_unverified": anchors,
            "source_anchor_excerpts_unverified": anchor_excerpts,
            "human_relevance_verdict": "unreviewed",
            "review_instructions": (
                "Independently inspect the cited policy excerpts and other potentially "
                "relevant corpus chunks. Anchor matches and retrieved hits are ONLY "
                "candidates, not correctness labels. Do not use the answer model output "
                "as ground truth. Recall requires sufficiently complete relevant-chunk labels."
            ),
        })
        label_rows.append({
            "question_id": qid,
            "split": "dev",
            "answerable": True,
            "source_dataset_sha256": digests["dataset"],
            "candidate_chunk_ids_unverified": list(dict.fromkeys(ranked_ids[:top_k] + anchor_ids)),
            "verified_relevant_chunk_ids": [],
            "review_state": "unreviewed",
            "reviewer": None,
            "review_notes": "",
        })

    output_dir.mkdir(parents=True, exist_ok=False)
    packet_path = output_dir / "retrieval_review_packet.jsonl"
    labels_path = output_dir / "relevance_labels_UNREVIEWED.jsonl"
    with packet_path.open("x", encoding="utf-8") as stream:
        for row in review_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with labels_path.open("x", encoding="utf-8") as stream:
        for row in label_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "split": "dev",
        "questions_exported": len(selected),
        "question_ids": [q["question_id"] for q in selected],
        "top_k_source_excerpts": top_k,
        "review_packets": str(packet_path),
        "blank_relevance_labels": str(labels_path),
        "source_dataset_sha256": digests["dataset"],
        "source_corpus_sha256": digests["corpus"],
        "source_retrieval_sha256": digests["retrieval"],
        "new_embedding_calls": 0,
        "new_qdrant_queries": 0,
        "new_generation_calls": 0,
        "verified_labels": 0,
        "warning": "This exports review CANDIDATES only; no Recall/MRR is certified.",
    }
