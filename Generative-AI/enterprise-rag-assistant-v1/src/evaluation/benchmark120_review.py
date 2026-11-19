"""Aggregate explicit human annotations; never assign semantic grades automatically."""
from __future__ import annotations

from collections import Counter
import math
from typing import Any

from src.evaluation.experiment import fingerprint
from src.generation.citation_builder import EVIDENCE_GROUP_PATTERN, EVIDENCE_ID_PATTERN


def review_template(question: dict, answer: dict) -> dict:
    return {
        "answer_sha256": fingerprint(answer), "review_state": "unreviewed",
        "reviewer": None, "reviewer_kind": None, "claims_exhaustive": False,
        "facts": [{"fact": fact, "importance": None, "status": None,
                   "evidence_ids": [], "notes": ""} for fact in question["expected_facts"]],
        "claims": [],  # {text, support: supported/unsupported/uncertain, evidence_ids}
        "citation_checks": [],  # {claim_index: 0-based, evidence_id, supports: bool}
        "answer_abstained": None, "abstention_correct": None, "notes": "",
    }


def summarize_reviews(questions: list[dict], answers: dict[str, dict], records: list[dict]) -> dict:
    known = {q["question_id"]:q for q in questions}
    seen, completed = set(), []
    for record in records:
        qid = record.get("question_id")
        if qid not in known or qid not in answers or qid in seen:
            raise ValueError("Review has duplicate, unknown, or uncaptured question")
        seen.add(qid)
        review = record.get("review", {})
        if review.get("answer_sha256") != fingerprint(answers[qid]):
            raise ValueError("Review is stale: answer or its evidence changed")
        if review.get("review_state") != "complete":
            continue
        if review.get("reviewer_kind") != "human":
            raise ValueError("Human quality metrics require an explicitly identified human reviewer")
        if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
            raise ValueError("Completed review requires reviewer name")
        if review.get("claims_exhaustive") is not True:
            raise ValueError("Reviewer must confirm all factual claims are annotated")
        facts = review.get("facts")
        if not isinstance(facts,list) or [f.get("fact") for f in facts] != known[qid]["expected_facts"]:
            raise ValueError("Fact review must preserve original expected facts")
        evidence_ids = {e["evidence_id"] for e in answers[qid]["selected_evidence"]}
        def valid_ids(ids):
            return (isinstance(ids,list) and all(type(i) is int and i in evidence_ids for i in ids)
                    and len(set(ids)) == len(ids))
        for fact in facts:
            if (fact.get("importance") not in {"required","optional"}
                or fact.get("status") not in {"met","partial","missed","contradicted"}
                or not valid_ids(fact.get("evidence_ids"))):
                raise ValueError("Invalid or incomplete fact annotation")
            if fact["importance"] == "optional" and not str(fact.get("notes", "")).strip():
                raise ValueError("Optional background facts require reviewer rationale")
        claims = review.get("claims")
        if not isinstance(claims,list):
            raise ValueError("Claims must be a list")
        for claim in claims:
            if (not isinstance(claim.get("text"),str) or not claim["text"].strip()
                or claim["text"] not in answers[qid]["answer"]
                or claim.get("support") not in {"supported","unsupported","uncertain"}
                or not valid_ids(claim.get("evidence_ids"))
                or (claim["support"] == "supported" and not claim["evidence_ids"])):
                raise ValueError("Invalid claim annotation: quote the answer and identify supporting evidence")
        checks = review.get("citation_checks")
        if not isinstance(checks,list):
            raise ValueError("Citation checks must be a list")
        expected_pairs = {(index,int(eid)) for index,claim in enumerate(claims)
                          for group in EVIDENCE_GROUP_PATTERN.findall(claim["text"])
                          for eid in EVIDENCE_ID_PATTERN.findall(group)}
        actual_cited_ids = {c["evidence_id"] for c in answers[qid].get("citations",[])}
        if {eid for _,eid in expected_pairs} != actual_cited_ids:
            raise ValueError("Claim quotations must include every actual answer citation")
        pairs = []
        for check in checks:
            if type(check.get("supports")) is not bool:
                raise ValueError("Each claim-to-citation check needs a support verdict")
            pairs.append((check.get("claim_index"),check.get("evidence_id")))
        if len(set(pairs)) != len(pairs) or set(pairs) != expected_pairs:
            raise ValueError("Citation checks must cover every annotated claim/evidence pair")
        if any(type(review.get(key)) is not bool for key in ("answer_abstained","abstention_correct")):
            raise ValueError("Abstention review must be explicit")
        completed.append(review)
    facts = [f for r in completed for f in r["facts"] if f["importance"] == "required"]
    claims = [c for r in completed for c in r["claims"]]
    citations = [c for r in completed for c in r["citation_checks"]]
    return {
        "human_reviews_completed":len(completed), "captured_answers":len(answers),
        "required_facts_reviewed":len(facts),
        "required_fact_completeness":sum(f["status"] == "met" for f in facts)/len(facts) if facts else None,
        "fact_status_counts":dict(Counter(f["status"] for f in facts)),
        "claims_reviewed":len(claims),
        "claim_citation_coverage":sum(bool(EVIDENCE_GROUP_PATTERN.search(c["text"])) for c in claims)/len(claims) if claims else None,
        "claim_groundedness":sum(c["support"] == "supported" for c in claims)/len(claims) if claims else None,
        "unsupported_claims":sum(c["support"] == "unsupported" for c in claims),
        "uncertain_claims":sum(c["support"] == "uncertain" for c in claims),
        "citation_pairs_reviewed":len(citations),
        "citation_correctness":sum(c["supports"] for c in citations)/len(citations) if citations else None,
        "abstention_accuracy":sum(r["abstention_correct"] for r in completed)/len(completed) if completed else None,
        "warning":"Human annotations only; partial reviews are not a complete benchmark. Labels are not independently authenticated by code.",
    }


def summarize_attempts(events: list[dict[str, Any]]) -> dict:
    started = {(e["run_id"],e["question_id"],e["attempt"]) for e in events if e["event"] == "started"}
    finished = [e for e in events if e["event"] in {"succeeded","failed"}]
    finished_ids = {(e["run_id"],e["question_id"],e["attempt"]) for e in finished}
    failed = [e for e in finished if e["event"] == "failed"]
    seconds = sorted(e["seconds"] for e in finished)
    return {
        "provider_attempts_started":len(started), "provider_attempts_finished":len(finished),
        "unfinished_attempts":len(started-finished_ids),
        "provider_error_rate":len(failed)/len(finished) if finished else None,
        "failure_categories":dict(Counter(e["category"] for e in failed)),
        "provider_latency_p50_seconds":seconds[math.ceil(len(seconds)*0.50)-1] if seconds else None,
        "provider_latency_p95_seconds":seconds[math.ceil(len(seconds)*0.95)-1] if seconds else None,
        "answer_failures":dict(Counter(e["category"] for e in events if e["event"] == "answer_failed")),
        "warning":"Request latency includes successful and failed provider attempts; excludes retrieval. Historical attempts without a ledger are unknown.",
    }
