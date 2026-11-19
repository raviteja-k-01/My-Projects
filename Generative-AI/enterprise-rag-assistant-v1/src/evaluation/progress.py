"""Offline progress and review material; never infer human quality scores."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.evaluation.answer_quality import summarize_reviews, validate_review
from src.evaluation.replay import is_replayable


STATE_DEFINITIONS = {
    "NOT STARTED": "No retrieval checkpoint or captured answer.",
    "RETRIEVAL COMPLETE": "Replayable retrieval saved; answer not captured.",
    "GENERATION COMPLETE": "An answer is saved; this is not a quality judgment.",
    "NEEDS REVIEW": "Answer saved; independent human review is incomplete or has open findings.",
    "VERIFIED": "Source mechanics pass and a complete human review gives all 2s, all facts met, no flags.",
    "FAILED": "Saved answer fails evidence/citation checks or a completed review flags a critical problem.",
    "BLOCKED": "Legacy skipped retrieval, missing sources, or an explicitly recorded retrieval blocker prevent progress.",
}


def build_progress(
    gold: list[dict[str, Any]], cases: dict[str, dict[str, Any]],
    reviews: dict[str, dict[str, Any]], checkpoints: dict[str, dict[str, Any]],
    chunks: dict[str, dict[str, Any]],
    *, retrieval_blocker: str | None = None,
) -> dict[str, Any]:
    summary = summarize_reviews(gold, cases, reviews)
    rows = []
    for record in gold:
        qid = record["question_id"]
        case, review, checkpoint = cases[qid], reviews[qid], checkpoints.get(qid)
        run = case["run"]
        expected = record["expected_chunk_ids"]
        missing = [cid for cid in expected if cid not in chunks]
        ranked = checkpoint["retrieved_chunk_ids"] if checkpoint else []
        selected = run["selected_evidence"] if run else []
        evidence_errors = []
        for item in selected:
            source = chunks.get(item["chunk_id"])
            if source is None or any(
                item.get(field) != (source.get(field, "").strip() if field == "text" else source.get(field))
                for field in ("text", "document_id", "page_start", "page_end")
            ):
                evidence_errors.append(item["chunk_id"])
        review_complete = validate_review(review, record)
        state = "NOT STARTED"
        reason = "Retrieval and generation have not run."
        if not checkpoint and not run and retrieval_blocker:
            state, reason = "BLOCKED", retrieval_blocker
        if checkpoint:
            if is_replayable(checkpoint):
                state, reason = "RETRIEVAL COMPLETE", "Saved retrieval is ready for answer capture."
            else:
                state, reason = "BLOCKED", "Legacy record skipped retrieval; empty evidence is not established."
        if run:
            state, reason = "NEEDS REVIEW", "Independent human quality review is incomplete."
            if evidence_errors or not case["mechanical_checks"]["mechanically_valid"]:
                state, reason = "FAILED", "Evidence or citation mechanics failed; inspect the saved answer."
            elif qid in summary["reviewed_problem_cases"]:
                state, reason = "FAILED", "Completed human review records a critical problem."
            elif review_complete:
                if (all(score == 2 for score in review["ratings"].values())
                        and all(fact["status"] == "met" for fact in review["fact_checks"])):
                    state, reason = "VERIFIED", "Human review and structural checks meet the documented gate."
                else:
                    reason = "Human review is complete but records partial coverage or other open findings."
        if missing:
            state, reason = "BLOCKED", "Gold source references are missing from the verified corpus."
        rows.append({
            "question_id": qid, "question_type": record["question_type"],
            "gold_answerable": record["answerable"],
            "expected_chunk_count": len(expected), "missing_expected_chunks": missing,
            "retrieval_checkpoint": "saved" if checkpoint else "missing",
            "retrieved_chunk_count": len(ranked) if checkpoint else None,
            "retrieval_provenance": ("recorded" if checkpoint and checkpoint.get("provenance")
                                     else "historical_unknown" if checkpoint else None),
            "expected_chunks_retrieved": [cid for cid in expected if cid in ranked],
            "expected_chunks_selected": [cid for cid in expected if cid in {e["chunk_id"] for e in selected}],
            "answer_status": "GENERATION COMPLETE" if run else "NOT STARTED",
            "mechanical_citations_valid": case["mechanical_checks"]["mechanically_valid"] if run else None,
            "evidence_matches_corpus": not evidence_errors if run else None,
            "evidence_errors": evidence_errors,
            "human_review": "complete" if review_complete else "incomplete",
            "status": state, "reason": reason,
        })
    return {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "retrieval_blocker": retrieval_blocker, "state_definitions": STATE_DEFINITIONS,
            "summary": summary, "questions": rows}


def format_review_packet(
    gold: list[dict[str, Any]], cases: dict[str, dict[str, Any]],
    checkpoints: dict[str, dict[str, Any]], chunks: dict[str, dict[str, Any]],
) -> str:
    lines = ["# Answer review packet", "",
             "Generated offline. No human grades or factual-correctness scores assigned.",
             "Exact source-ID coverage is diagnostic; equivalent facts may occur in other chunks."]
    for record in gold:
        qid = record["question_id"]
        run = cases[qid]["run"]
        lines += ["", f"## {qid}: {record['question']}", "",
                  f"Gold answerable: {record['answerable']} (evaluation label only).",
                  f"Saved ranking: {checkpoints.get(qid, {}).get('retrieved_chunk_ids', 'not captured')}",
                  "", "Expected facts:", ""]
        lines += [f"- {fact}" for fact in record["expected_facts"]]
        lines += ["", "Captured answer:", "", run["answer"] if run else "NOT CAPTURED."]
        if run:
            lines += ["", f"Model: {run.get('model')}; retrieval source: {run.get('retrieval_source', 'unknown')}.",
                      "", "Actual evidence supplied to generation:"]
            for evidence in run["selected_evidence"]:
                lines += ["", f"### Evidence {evidence['evidence_id']}: {evidence['chunk_id']}",
                          f"Document: {evidence.get('document_id')}; pages: {evidence.get('page_start')}-{evidence.get('page_end')}",
                          "", evidence["text"]]
        lines += ["", "Gold-referenced sources (may differ from selected evidence):"]
        for cid in record["expected_chunk_ids"]:
            source = chunks[cid]
            lines += ["", f"### {cid}",
                      f"Document: {source.get('document_id')}; pages: {source.get('page_start')}-{source.get('page_end')}",
                      "", source["text"]]
        lines += ["", "Reviewer: assess facts, unsupported claims, relevance, citations and abstention separately.",
                  f"Record actual judgments in answer_review_forms/{qid}.json; never treat retrieval rank as correctness."]
    return "\n".join(lines) + "\n"
