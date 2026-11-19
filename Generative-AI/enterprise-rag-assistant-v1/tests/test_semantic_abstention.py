"""Contract and limitation tests, NOT evidence of a model's semantic accuracy."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.rag.rag_pipeline import EnterpriseRAGPipeline
from src.generation.answer_generator import GroundedAnswerGenerator
from src.generation.citation_builder import CitationBuilder
from src.retrieval.context_builder import ContextBuilder

CASES = json.loads((Path(__file__).parent / "fixtures/insufficient_evidence.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
@pytest.mark.parametrize("source", ["live", "saved_hybrid_checkpoint"])
@pytest.mark.parametrize("fabricated", [False, True], ids=["compliant-abstention", "known-semantic-gap"])
def test_nonempty_insufficient_evidence_contract_and_known_gap(case, source, fabricated):
    from src.rag.answer_composer import compose_answer
    chunk = case["chunk"]
    candidates = [{"rank": 1, "chunk_id": chunk["chunk_id"], "chunk": chunk}]
    unsupported = {
        "Q026": "Every regulator must be notified within exactly 72 hours. [Evidence 1]",
        "Q027": "The exact fine will be 123456 dollars. [Evidence 1]",
        "Q028": "That client requires biometric login for every contractor. [Evidence 1]",
    }
    answer = (unsupported[case["id"]] if fabricated else
              "The supplied evidence is insufficient to determine the requested exact details.")
    calls = []
    generator = GroundedAnswerGenerator.__new__(GroundedAnswerGenerator)
    generator.model_name = "mock-only"

    def invoke(variables):
        assert set(variables) == {"question", "context"}
        assert variables["question"] == case["question"]
        assert chunk["text"] in variables["context"]
        calls.append(variables)
        return SimpleNamespace(text=answer, usage_metadata={}, response_metadata={"finish_reason": "STOP"})

    generator.chain = SimpleNamespace(invoke=invoke)
    pipeline = EnterpriseRAGPipeline.__new__(EnterpriseRAGPipeline)
    pipeline.retriever = SimpleNamespace(retrieve_with_mode=lambda **kw: (candidates, "hybrid"))
    pipeline.generator = generator
    pipeline.context_builder = ContextBuilder()
    pipeline.citation_builder = CitationBuilder()
    if source == "live":
        result = pipeline.answer(case["question"], evaluation_trace=True)
    else:
        result = compose_answer(question=case["question"], candidates=candidates,
            retrieval_mode="hybrid", retrieval_source=source, generator=generator,
            context_builder=pipeline.context_builder, citation_builder=pipeline.citation_builder,
            retrieval_seconds=None, evaluation_trace=True)
    assert len(calls) == 1
    assert result["answer"] == answer
    assert result["evidence_count"] == 1
    if fabricated:
        # Deliberately document the unresolved gap: a real ID cannot establish
        # entailment. Passing this test must NEVER be counted as safe abstention.
        assert result["citations"][0]["chunk_id"] == chunk["chunk_id"]
    else:
        assert result["citations"] == []
