"""Evidence-first, human-reviewed RAG answer-quality evaluation.

Automated checks validate citation *mechanics*, never factual support.
Judging factual correctness, completeness, claim-to-source alignment,
relevance and clarity requires a reviewer reading the actual evidence.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from src.generation.citation_builder import (
    EVIDENCE_GROUP_PATTERN,
    EVIDENCE_ID_PATTERN,
)

DIMENSIONS = (
    "evidence_sufficiency",
    "answer_relevance",
    "factual_grounding",
    "fact_coverage",
    "citation_support",
    "structure_clarity",
    "uncertainty_calibration",
)
FACT_STATUSES = frozenset({"met", "partial", "missed", "contradicted"})


def new_case(gold: dict[str, Any]) -> dict[str, Any]:
    """Gold rubric anchor. An empty run is NOT a failed answer."""
    return {
        "question_id": gold["question_id"],
        "question": gold["question"],
        "question_type": gold["question_type"],
        "answerable": gold["answerable"],
        "expected_facts": list(gold["expected_facts"]),
        "expected_chunk_ids": list(gold["expected_chunk_ids"]),
        "run": None,
        "mechanical_checks": None,
    }


def new_review(gold: dict[str, Any]) -> dict[str, Any]:
    """Null grades are deliberate: no model may self-award high scores."""
    return {
        "question_id": gold["question_id"],
        "reviewer": None,
        "notes": "",
        "ratings": {dimension: None for dimension in DIMENSIONS},
        "fact_checks": [
            {"fact": fact, "status": None, "notes": ""}
            for fact in gold["expected_facts"]
        ],
        "unsupported_claims": [],
        "citation_errors": [],
    }


def audit_citation_mechanics(answer: str, citations: list[dict[str, Any]],
                             evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Check syntax and metadata against the supplied evidence; NOT support.

    A citation like [Evidence 1] may be real yet fail to support the claim.
    That requires human citation_support review.
    """
    evidence_map = {record["evidence_id"]: record for record in evidence}
    if len(evidence_map) != len(evidence):
        raise ValueError("Duplicate evidence ID in evaluation trace")
    citation_map = {record["evidence_id"]: record for record in citations}
    if len(citation_map) != len(citations):
        raise ValueError("Duplicate mapped citation in answer")

    groups = EVIDENCE_GROUP_PATTERN.findall(answer)
    cited_ids = {
        int(number)
        for group in groups
        for number in EVIDENCE_ID_PATTERN.findall(group)
    }
    # Recognize malformed/extraneous Evidence N references outside valid brackets.
    all_mentions = {
        int(number) for number in EVIDENCE_ID_PATTERN.findall(answer)
    }
    unknown_mentions = sorted(all_mentions - set(evidence_map))
    mismatches = []
    for evidence_id, source in citation_map.items():
        if evidence_id not in evidence_map:
            mismatches.append(f"unknown_citation:{evidence_id}")
            continue
        expected = evidence_map[evidence_id]
        if source.get("chunk_id") != expected.get("chunk_id"):
            mismatches.append(f"chunk_mismatch:{evidence_id}")
        if source.get("document_id") != expected.get("document_id"):
            mismatches.append(f"document_mismatch:{evidence_id}")

    if cited_ids != set(citation_map):
        mismatches.append("citation_ids_do_not_match_text")
    if all_mentions != cited_ids:
        mismatches.append("malformed_or_unbracketed_evidence_reference")
    return {
        "has_citations": bool(cited_ids),
        "citation_ids": sorted(cited_ids),
        "unknown_evidence_mentions": unknown_mentions,
        "metadata_mismatches": mismatches,
        "mechanically_valid": not unknown_mentions and not mismatches,
        "NOTE": "Mechanical validity does not establish factual grounding.",
    }


def attach_capture(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    """Capture exactly the run returned by the real RAG pipeline."""
    if case["run"] is not None:
        raise ValueError("Refusing to overwrite an existing answer capture")
    if response.get("question") != case["question"].strip():
        raise ValueError("Captured question does not match gold question")
    if response.get("retrieval_mode") != "hybrid":
        raise ValueError("Strict evaluation requires hybrid retrieval")
    trace = response.get("evaluation_trace")
    if not isinstance(trace, dict) or not isinstance(trace.get("selected_evidence"), list):
        raise ValueError("Evaluation trace with original evidence text is missing")
    if not isinstance(response.get("answer"), str) or not response["answer"].strip():
        raise ValueError("Captured answer is missing")
    if not isinstance(response.get("citations"), list):
        raise ValueError("Captured citation list is missing")
    if response.get("citation_count") != len(response["citations"]):
        raise ValueError("Citation count does not match citation list")
    if not isinstance(trace.get("retrieved_chunk_ids"), list):
        raise ValueError("Retrieved candidate IDs are missing from trace")

    check = audit_citation_mechanics(
        response["answer"], response["citations"], trace["selected_evidence"]
    )
    copied = dict(case)
    copied["run"] = {
        "answer": response["answer"],
        "model": response.get("model"),
        "retrieval_mode": response["retrieval_mode"],
        "citations": response["citations"],
        "retrieved_chunk_ids": trace["retrieved_chunk_ids"],
        "selected_evidence": trace["selected_evidence"],
        "latency": response.get("latency"),
        "usage": response.get("usage"),
    }
    copied["mechanical_checks"] = check
    return copied


def validate_review(review: dict[str, Any], gold: dict[str, Any]) -> bool:
    """Validate review fields; return True only for fully reviewed cases.

    An unreviewed/partially reviewed record is legitimate and not scored.
    """
    if review.get("question_id") != gold["question_id"]:
        raise ValueError("Review question ID does not match gold")
    checks = review.get("fact_checks")
    if not isinstance(checks, list) or [c.get("fact") for c in checks] != gold["expected_facts"]:
        raise ValueError(f"Gold facts changed or review facts missing for {gold['question_id']}")
    ratings = review.get("ratings")
    if not isinstance(ratings, dict) or set(ratings) != set(DIMENSIONS):
        raise ValueError(f"Invalid rating dimensions for {gold['question_id']}")
    if not isinstance(review.get("notes"), str):
        raise ValueError("Review notes must be text")
    if not isinstance(review.get("unsupported_claims"), list) or not all(
        isinstance(item, str) for item in review["unsupported_claims"]
    ):
        raise ValueError("unsupported_claims must contain text descriptions")
    if not isinstance(review.get("citation_errors"), list) or not all(
        isinstance(item, str) for item in review["citation_errors"]
    ):
        raise ValueError("citation_errors must contain text descriptions")
    for field in ratings.values():
        if field is not None and (type(field) is not int or field not in (0, 1, 2)):
            raise ValueError("Ratings must be integers 0, 1, 2, or null")
    for check in checks:
        if check.get("status") is not None and check["status"] not in FACT_STATUSES:
            raise ValueError("Fact status must be met, partial, missed, contradicted, or null")
        if not isinstance(check.get("notes"), str):
            raise ValueError("Fact notes must be text")
    complete = (
        isinstance(review.get("reviewer"), str)
        and bool(review["reviewer"].strip())
        and all(score is not None for score in ratings.values())
        and all(check.get("status") is not None for check in checks)
    )
    return complete


def summarize_reviews(gold: list[dict[str, Any]], cases: dict[str, dict[str, Any]],
                      reviews: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_id = {record["question_id"]: record for record in gold}
    if set(cases) != set(by_id) or set(reviews) != set(by_id):
        raise ValueError("Evaluation file question IDs do not match the gold dataset")

    captured = 0
    complete = []
    mechanically_invalid = []
    for question_id, reference in by_id.items():
        case = cases[question_id]
        if case["run"] is not None:
            captured += 1
            if not case["mechanical_checks"]["mechanically_valid"]:
                mechanically_invalid.append(question_id)
        if validate_review(reviews[question_id], reference):
            if case["run"] is None:
                raise ValueError(f"Review completed without answer capture: {question_id}")
            complete.append(question_id)

    scores = {dimension: dict(sorted(Counter(
        reviews[question_id]["ratings"][dimension] for question_id in complete
    ).items())) for dimension in DIMENSIONS}
    fact_statuses = dict(Counter(
        item["status"]
        for question_id in complete
        for item in reviews[question_id]["fact_checks"]
    ))
    problems = [
        question_id for question_id in complete
        if reviews[question_id]["unsupported_claims"]
        or reviews[question_id]["citation_errors"]
        or any(score == 0 for score in reviews[question_id]["ratings"].values())
        or any(item["status"] == "contradicted" for item in reviews[question_id]["fact_checks"])
    ]
    return {
        "gold_questions": len(gold),
        "answerable_questions": sum(row["answerable"] for row in gold),
        "unanswerable_questions": sum(not row["answerable"] for row in gold),
        "captured": captured,
        "fully_human_reviewed": len(complete),
        "unreviewed_or_partial": len(gold) - len(complete),
        "mechanical_citation_failures": mechanically_invalid,
        "ratings_distribution": scores,
        "expected_fact_statuses": fact_statuses,
        "reviewed_problem_cases": problems,
        "note": "No aggregate RAG score. No inferred correctness for unreviewed answers.",
    }
