"""Offline regression for the Q030 failure class, with no provider requests."""
from src.retrieval.comparison_coverage import comparison_facets, cover_comparison_facets
from src.retrieval.context_builder import ContextBuilder

QUESTION = ("Are AI vendor risk tier, AI use-case governance risk tier, "
            "and security-incident severity the same classification system? "
            "Explain the difference.")


def entry(cid, doc, name, text, *, rank=1):
    return {"chunk_id":cid, "rank":rank, "score":1.0, "chunk":{
        "chunk_id":cid, "document_id":doc, "document_name":name,
        "section_title":"Risk tiers and classification", "text":text,
        "page_start":1, "page_end":1, "token_count":20}}


AI = entry("AI1", "AI-GOV-05", "AI Governance Standard",
           "AI use case governance risk tiers classify autonomy data and potential impact.")
VENDOR = entry("VEN1", "PROC-TPRM-11", "Procurement Vendor Risk Policy",
               "Third-party vendor risk tier categories drive supplier due diligence and treatment.")
INCIDENT = entry("IR1", "SEC-IR-004", "Security Incident Response Manual",
                 "Security incident severity is classified from SEV-1 through SEV-4.")


def test_extracts_comparison_facets_but_not_regular_questions():
    assert comparison_facets(QUESTION) == [
        "AI vendor risk tier", "AI use-case governance risk tier", "security-incident severity"]
    assert comparison_facets("What are the requirements for password and MFA access?") == []
    assert comparison_facets("Is a password secure?") == []


def test_facet_coverage_exposes_three_distinct_document_sources():
    baseline = [AI] + [entry(f"AI{i}", "AI-GOV-05", "AI Governance Standard",
               f"AI governance tier guidance {i}.", rank=i) for i in range(2, 21)]
    searches = []
    def bm25(facet, limit):
        searches.append(facet)
        if "vendor" in facet:
            return [VENDOR]
        if "security-incident" in facet:
            return [INCIDENT]
        return [AI]
    ranked = cover_comparison_facets(baseline, comparison_facets(QUESTION), bm25, limit=20)
    assert searches == comparison_facets(QUESTION)
    assert [c["chunk"]["document_id"] for c in ranked[:3]] == [
        "AI-GOV-05", "PROC-TPRM-11", "SEC-IR-004"]
    assert [c["rank"] for c in ranked] == list(range(1, 21))
    assert ranked[1]["facet_query"] == "AI vendor risk tier"
    assert ranked[2]["rrf_score"] is None
    evidence = ContextBuilder(max_chunks=8).build(ranked)["evidence"]
    assert {e["document_id"] for e in evidence} == {
        "AI-GOV-05", "PROC-TPRM-11", "SEC-IR-004"}


def test_no_silent_invention_when_facet_evidence_absent():
    missing = cover_comparison_facets([AI], comparison_facets(QUESTION),
                                      lambda _f, _n: [], limit=20)
    assert [i["chunk_id"] for i in missing] == ["AI1"]


def test_does_not_promote_irrelevant_document():
    irrelevant = entry("X", "HR-LV-07", "Leave and Attendance", "Marriage leave is five days")
    result = cover_comparison_facets([AI], comparison_facets(QUESTION),
                                     lambda _f, _n: [irrelevant], limit=20)
    assert [i["chunk_id"] for i in result] == ["AI1"]


def test_does_not_modify_original_candidate_ranks():
    baseline = [dict(AI), dict(VENDOR)]
    before = [x["rank"] for x in baseline]
    cover_comparison_facets(baseline, comparison_facets(QUESTION),
                           lambda _f, _n: [INCIDENT], limit=20)
    assert [x["rank"] for x in baseline] == before


def test_diagnostic_readers_accept_real_checkpoint_shape(tmp_path):
    import json
    from scripts.check_comparison_coverage import _load_case, _load_chunks
    checkpoint = tmp_path / 'saved.jsonl'
    chunks = tmp_path / 'chunks.jsonl'
    checkpoint.write_text(json.dumps({"question_id":"Q030", "question":QUESTION,
                    "retrieved_chunk_ids":["AI1"]})+'\n')
    chunks.write_text(json.dumps(AI['chunk'])+'\n')
    case = _load_case(checkpoint, 'Q030')
    loaded = _load_chunks(chunks, set(case['retrieved_chunk_ids']))
    assert loaded['AI1']['document_id'] == 'AI-GOV-05'


def test_severity_facet_prefers_explicit_levels_over_generic_triage():
    generic = entry("IR-TRIAGE", "SEC-IR-004", "Incident Manual",
                    "Initial incident severity governs response triage and containment.")
    defined = entry("IR-LEVELS", "SEC-IR-004", "Incident Manual",
                    "Incident severity uses SEV-1 through SEV-4 classifications.")
    seen_queries = []
    def sparse(query, _limit):
        seen_queries.append(query)
        if query == "security-incident severity":
            return [generic]
        if query == "incident severity classification SEV-1 SEV-4":
            return [defined]
        return [AI]
    selected = cover_comparison_facets([AI], comparison_facets(QUESTION), sparse, limit=8)
    assert selected[1]["chunk_id"] == "IR-LEVELS"
    assert "incident severity classification SEV-1 SEV-4" in seen_queries


def test_severity_facet_does_not_invent_levels_if_search_lacks_them():
    generic = entry("IR-TRIAGE", "SEC-IR-004", "Incident Manual",
                    "Initial incident severity governs response triage and containment.")
    def sparse(query, _limit):
        if query == "security-incident severity":
            return [generic]
        return [AI]
    selected = cover_comparison_facets([AI], comparison_facets(QUESTION), sparse, limit=8)
    assert selected[1]["chunk_id"] == "IR-TRIAGE"
