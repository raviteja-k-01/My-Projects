from types import SimpleNamespace

import pytest
from fastapi import Response
from src.retrieval.context_builder import ContextBuilder
from src.generation.citation_builder import CitationBuilder
from src.rag.answer_composer import compose_answer


def candidate(cid, text):
    return {"chunk_id":cid, "rank":1, "chunk":{"chunk_id":cid,"text":text,"token_count":20}}


@pytest.mark.parametrize("other", [
    "Employees must not submit a request within 30 days after the travel is completed.",
    "Employees must submit a request within 60 days after the travel is completed.",
])
def test_preserve_conflicting_near_identical_policies(other):
    original = "Employees must submit a request within 30 days after the travel is completed."
    out = ContextBuilder().build([candidate("A",original), candidate("B",other)])
    assert out["evidence_count"] == 2


def test_refill_evidence_after_duplicates():
    items = [candidate(str(i),"Same policy wording") for i in range(8)]
    items.append(candidate("ninth", "A different policy condition."))
    contexts = []
    def generate(**kw):
        contexts.append(kw["context"])
        return {"answer":"Condition [Evidence 2]","model":"fake","usage":{}}
    out = compose_answer(question="Question", candidates=items, retrieval_mode="hybrid",
        retrieval_source="live", retrieval_seconds=0, context_builder=ContextBuilder(),
        generator=SimpleNamespace(generate=generate), citation_builder=CitationBuilder())
    assert out["evidence_count"] == 2 and out["citations"][0]["chunk_id"] == "ninth"


def test_small_real_bm25_index_excludes_nonmatches(tmp_path):
    import bm25s
    from bm25s.tokenization import Tokenizer
    from src.indexing.bm25_index import BM25Index, split_text
    index = BM25Index(tmp_path)
    index.tokenizer = Tokenizer(splitter=split_text)
    texts = ["travel expense", "security policy"]
    index.retriever = bm25s.BM25(corpus=[{"text":t} for t in texts])
    index.retriever.index(index.tokenizer.tokenize(texts,return_as="tuple"))
    assert index.search("zzzzunknownvocabulary", limit=20) == []
    results = index.search("travel", limit=20)
    assert len(results) == 1 and results[0]["chunk"]["text"] == texts[0]


@pytest.mark.parametrize("exc", [RuntimeError("secret-key"),ValueError("secret-url"),OSError("secret-path")])
def test_api_sanitizes_internal_errors(monkeypatch, exc):
    from src.api import app as module
    def fail(**kw): raise exc
    monkeypatch.setattr(module,"rag_pipeline",SimpleNamespace(answer=fail))
    with pytest.raises(module.HTTPException) as caught:
        module.ask(module.AskRequest(question="Question"), Response())
    assert caught.value.status_code >= 500
    assert "secret" not in caught.value.detail


def test_duplicate_rrf_source_entry_cannot_inflate_rank():
    from src.retrieval.rrf import reciprocal_rank_fusion
    row = {"chunk_id":"A","rank":1,"score":1,"chunk":{"text":"A"}}
    with pytest.raises(ValueError,match="Duplicate"):
        reciprocal_rank_fusion([row,row],[])


@pytest.mark.parametrize("text", ["Claim [Evidence one]", "Claim [Evidence 1]. Evidence 1"])
def test_invalid_citations_match_live_and_audit_rejection(text):
    from src.evaluation.answer_quality import audit_citation_mechanics
    evidence = [{"evidence_id":1,"chunk_id":"A"}]
    with pytest.raises(ValueError,match="Malformed"):
        CitationBuilder().build(text,evidence)
    assert not audit_citation_mechanics(text,[],evidence)["mechanically_valid"]
