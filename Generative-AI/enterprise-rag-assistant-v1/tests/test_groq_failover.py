"""Offline Groq failover contract tests; no provider calls."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.generation.failover_generator import FailoverAnswerGenerator, GroqAnswerGenerator
from src.generation.reliability import GenerationError


class DummyPrimary:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, 0

    def generate(self, question, context):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


class DummyFallback:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, 0

    def generate(self, question, context):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


@pytest.mark.parametrize("category", ["unavailable", "timeout", "server", "transport", "quota"])
def test_transient_gemini_failure_switches_to_groq(category):
    primary = DummyPrimary(error=GenerationError(category, "primary unavailable", code=503))
    secondary = DummyFallback(result={"answer": "Yes [Evidence 1]", "model": "openai/gpt-oss-120b", "usage": {}})
    answer = FailoverAnswerGenerator(primary, secondary).generate("Question", "[Evidence 1] source")
    assert (primary.calls, secondary.calls) == (1, 1)
    assert answer["model"] == "openai/gpt-oss-120b"
    assert answer["generation_fallback_used"] is True


def test_primary_success_never_calls_groq():
    primary = DummyPrimary(result={"answer": "Yes", "model": "gemini-test", "usage": {}})
    secondary = DummyFallback(result=None)
    answer = FailoverAnswerGenerator(primary, secondary).generate("q", "e")
    assert answer["model"] == "gemini-test"
    assert secondary.calls == 0
    assert "generation_fallback_used" not in answer


@pytest.mark.parametrize("category", ["authentication", "request", "invalid_response", "budget_exhausted"])
def test_non_transient_error_fails_closed(category):
    primary = DummyPrimary(error=GenerationError(category, "unsafe primary failure"))
    secondary = DummyFallback(result=None)
    with pytest.raises(GenerationError) as exc:
        FailoverAnswerGenerator(primary, secondary).generate("q", "e")
    assert exc.value.category == category
    assert secondary.calls == 0


def test_groq_receives_same_question_evidence_and_prompt(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-placeholder-secret")
    monkeypatch.delenv("GROQ_FALLBACK_MODEL", raising=False)
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(status_code=200, json=lambda: {
            "choices": [{"message": {"content": "Policy [Evidence 1]"}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 15},
        })

    monkeypatch.setattr("src.generation.failover_generator.requests.post", fake_post)
    answer = GroqAnswerGenerator(system_prompt="Use evidence").generate(
        "Why?", "[Evidence 1] The policy says X."
    )
    url, opts = calls[0]
    assert url == "https://api.groq.com/openai/v1/chat/completions"
    assert opts["json"]["messages"][0]["content"] == "Use evidence"
    assert "[Evidence 1] The policy says X." in opts["json"]["messages"][1]["content"]
    assert "Why?" in opts["json"]["messages"][1]["content"]
    assert opts["json"]["model"] == "openai/gpt-oss-120b"
    assert opts["json"]["reasoning_effort"] == "low"
    assert opts["json"]["max_completion_tokens"] == 2048
    assert answer == {"answer": "Policy [Evidence 1]", "model": "openai/gpt-oss-120b", "usage": {"total_tokens": 15}}


@pytest.mark.parametrize("status,category", [(429, "quota"), (503, "unavailable"), (401, "authentication"), (404, "request")])
def test_groq_statuses_are_sanitized(monkeypatch, status, category):
    monkeypatch.setenv("GROQ_API_KEY", "secret")
    monkeypatch.setattr("src.generation.failover_generator.requests.post", lambda *a, **k:
                        SimpleNamespace(
    status_code=status,
    text="secret-details",
    headers={}
))
    with pytest.raises(GenerationError) as exc:
        GroqAnswerGenerator(system_prompt="Ground responses").generate("q", "e")
    assert exc.value.category == category
    assert "secret" not in str(exc.value)


@pytest.mark.parametrize("finish,text", [("length", "Partial [Evidence 1]"), ("stop", ""), ("stop", None)])
def test_groq_incomplete_or_missing_answer_fails_closed(monkeypatch, finish, text):
    monkeypatch.setenv("GROQ_API_KEY", "secret")
    monkeypatch.setattr("src.generation.failover_generator.requests.post", lambda *a, **k:
                        SimpleNamespace(status_code=200, json=lambda: {
                            "choices": [{"message": {"content": text}, "finish_reason": finish}]
                        }))
    with pytest.raises(GenerationError, match="complete answer"):
        GroqAnswerGenerator(system_prompt="Ground responses").generate("q", "e")


def test_groq_key_is_required_when_enabled(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        GroqAnswerGenerator(system_prompt="Ground responses")
