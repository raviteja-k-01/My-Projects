"""No-network tests: RAG-120 Groq generation uses saved strict-hybrid context."""
from __future__ import annotations

import json

import pytest

from src.evaluation import groq_answer_generator as module
from src.evaluation.benchmark120_answer_replay import run, report
from test_benchmark120_answer_replay import sample


class FakeGroqClient:
    def __init__(self, *, system_prompt):
        assert "Evidence" in system_prompt
        self.model_name = "openai/gpt-oss-120b"
        self.max_output_tokens = 2048
        self.timeout_seconds = 35
        self.calls = []

    def generate(self, question, context):
        assert "never put this in prompt" not in context
        assert "chunk B" in context
        self.calls.append((question, context))
        return {
            "answer": "Based on the policy [Evidence 1]",
            "model": self.model_name, "usage": {"total_tokens": 5},
        }


def test_groq_capture_is_source_locked_metered_and_resumable(monkeypatch, tmp_path):
    clients = []
    def create_client(*, system_prompt):
        client = FakeGroqClient(system_prompt=system_prompt)
        clients.append(client)
        return client
    monkeypatch.setattr(module, "GroqAnswerGenerator", create_client)
    opts = sample(tmp_path)
    outcome = run(**opts, provider="groq", max_new=1,
                  confirm_generation=True, request_budget=1)
    assert outcome["newly_saved"] == 1
    assert outcome["new_embedding_calls"] == 0
    assert outcome["new_qdrant_queries"] == 0
    assert outcome["provider_requests_used"] == 1
    assert len(clients) == 1 and len(clients[0].calls) == 1
    saved = json.loads(opts["answers_path"].read_text().splitlines()[0])
    assert saved["model"] == "openai/gpt-oss-120b"
    assert saved["experiment"]["generation"]["provider"] == "groq"
    assert saved["experiment"]["generation"]["max_attempts"] == 1
    assert saved["retrieval_source"] == "saved_hybrid_checkpoint"
    assert saved["citations"][0]["chunk_id"] == "B"
    assert "expected_facts" not in clients[0].calls[0][1]
    assert report(**opts)["answer_checkpointed"] == 1
    resumed = run(**opts, provider="groq", max_new=1,
                  confirm_generation=True, request_budget=1)
    assert resumed["newly_saved"] == 0
    assert resumed["answer_checkpointed"] == 1
    assert len(clients[0].calls) == 1


def test_groq_capture_rejects_model_mixing_before_api(monkeypatch, tmp_path):
    clients = []
    def create_client(*, system_prompt):
        client = FakeGroqClient(system_prompt=system_prompt)
        clients.append(client)
        return client
    monkeypatch.setattr(module, "GroqAnswerGenerator", create_client)
    opts = sample(tmp_path)
    with pytest.raises(ValueError, match="GROQ_FALLBACK_MODEL"):
        run(**opts, provider="groq", max_new=1, confirm_generation=True,
            model_name="other-groq-model", request_budget=1)
    assert not clients[0].calls


def test_live_generation_requires_confirmation_even_for_groq(tmp_path):
    opts = sample(tmp_path)
    with pytest.raises(ValueError, match="confirm-generation"):
        run(**opts, provider="groq", max_new=1)
    assert not opts["answers_path"].exists()


def test_bad_provider_rejected_offline(tmp_path):
    opts = sample(tmp_path)
    with pytest.raises(ValueError, match="provider"):
        run(**opts, provider="unknown")


def test_groq_adapter_sanitizes_attempt_event_and_counts_failure(monkeypatch):
    class ErrorClient(FakeGroqClient):
        def generate(self, question, context):
            from src.generation.reliability import GenerationError
            raise GenerationError("quota", "fake provider quota")
    monkeypatch.setattr(module, "GroqAnswerGenerator", ErrorClient)
    from src.generation.reliability import RequestBudget, GenerationError
    generator = module.EvaluatedGroqAnswerGenerator()
    generator.request_budget = RequestBudget(1)
    events = []
    generator.observer = events.append
    with pytest.raises(GenerationError, match="quota"):
        generator.generate("Question", "Evidence")
    assert generator.request_budget.used == 1
    assert [e["event"] for e in events] == ["started", "failed"]
    assert events[-1]["category"] == "quota"
    assert "fake provider quota" not in str(events)
