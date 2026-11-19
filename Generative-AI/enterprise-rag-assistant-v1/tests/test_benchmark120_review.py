from copy import deepcopy

import pytest

from src.evaluation.benchmark120_review import review_template, summarize_reviews, summarize_attempts


def fixture():
    q = {"question_id":"x","expected_facts":["Required fact", "Background"]}
    a = {"answer":"Supported statement. [Evidence 1] Invented amount.","selected_evidence":[{"evidence_id":1}],"citations":[{"evidence_id":1}]}
    r = review_template(q,a)
    return q,a,{"question_id":"x","review":r}


def complete():
    q,a,r = fixture()
    r["review"].update(review_state="complete",reviewer="Test human",reviewer_kind="human",
        claims_exhaustive=True,answer_abstained=False,abstention_correct=True,
        facts=[{"fact":"Required fact","importance":"required","status":"met","evidence_ids":[1]},
               {"fact":"Background","importance":"optional","status":"missed","evidence_ids":[],"notes":"Outside question scope"}],
        claims=[{"text":"Supported statement. [Evidence 1]","support":"supported","evidence_ids":[1]},
                {"text":"Invented amount.","support":"unsupported","evidence_ids":[]}],
        citation_checks=[{"claim_index":0,"evidence_id":1,"supports":True}])
    return q,a,r


def test_unreviewed_metrics_are_null():
    q,a,r = fixture()
    summary = summarize_reviews([q],{"x":a},[r])
    assert summary["human_reviews_completed"] == 0
    assert summary["claim_groundedness"] is None and summary["abstention_accuracy"] is None


def test_required_facts_claims_and_citation_metrics_have_explicit_denominators():
    q,a,r = complete()
    summary = summarize_reviews([q],{"x":a},[r])
    assert summary["required_facts_reviewed"] == 1
    assert summary["required_fact_completeness"] == 1
    assert summary["claims_reviewed"] == 2 and summary["claim_groundedness"] == 0.5
    assert summary["unsupported_claims"] == 1
    assert summary["citation_correctness"] == 1 and summary["abstention_accuracy"] == 1


@pytest.mark.parametrize("field,value",[("reviewer_kind","model"),("reviewer",None),
    ("claims_exhaustive",False),("abstention_correct",None),("answer_sha256","stale"),
    ("citation_checks",[])])
def test_unverifiable_or_incomplete_reviews_are_not_scored(field,value):
    q,a,r = complete()
    r["review"][field] = value
    with pytest.raises(ValueError):
        summarize_reviews([q],{"x":a},[r])


def test_changed_gold_fact_is_not_silently_accepted():
    q,a,r = complete()
    r["review"]["facts"][0]["fact"] = "Easier fact"
    with pytest.raises(ValueError,match="preserve"):
        summarize_reviews([q],{"x":a},[r])


def test_operations_include_failed_and_interrupted_requests():
    common = {"run_id":"r","question_id":"q"}
    events = [{**common,"attempt":i,"event":"started"} for i in (1,2,3)]
    events.extend([{**common,"attempt":1,"event":"failed","category":"timeout","seconds":10},
                   {**common,"attempt":2,"event":"succeeded","seconds":2}])
    summary = summarize_attempts(events)
    assert summary["unfinished_attempts"] == 1
    assert summary["provider_error_rate"] == 0.5
    assert summary["provider_latency_p95_seconds"] == 10
