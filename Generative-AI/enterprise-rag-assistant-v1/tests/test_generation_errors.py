"""Gemini generation failures are accurately classified without API calls."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.generation import answer_generator


class FakeServerError(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(f"fake provider status {code}")


class RaisingChain:
    def __init__(self, error):
        self.error = error
        self.calls = 0

    def invoke(self, variables):
        self.calls += 1
        assert variables["question"] == "What must be recorded?"
        assert variables["context"] == "[Evidence 1] ..."
        raise self.error


def make_generator(chain):
    generator = answer_generator.GroundedAnswerGenerator.__new__(
        answer_generator.GroundedAnswerGenerator
    )
    generator.chain = chain
    generator.model_name = "test-generation-model"
    return generator


@pytest.mark.parametrize("code, expected", [
    (503, "HTTP 503, high demand"),
    (504, "HTTP 504"),
    (500, "HTTP 500"),
    (502, "HTTP 502"),
])
def test_server_failure_is_reported_cleanly_without_extra_retries(
    monkeypatch, code, expected
):
    monkeypatch.setattr(answer_generator.errors, "ServerError", FakeServerError)
    chain = RaisingChain(FakeServerError(code))
    generator = make_generator(chain)
    with pytest.raises(RuntimeError, match=expected) as exc:
        generator.generate(
            question="What must be recorded?",
            context="[Evidence 1] ...",
        )
    assert "No answer" in str(exc.value)
    assert chain.calls == 1


def test_unknown_server_failure_propagates_instead_of_mislabeling_as_quota(monkeypatch):
    monkeypatch.setattr(answer_generator.errors, "ServerError", FakeServerError)
    with pytest.raises(FakeServerError) as exc:
        make_generator(RaisingChain(FakeServerError(599))).generate(
            question="What must be recorded?",
            context="[Evidence 1] ...",
        )
    assert exc.value.code == 599


def test_generation_model_is_explicitly_configurable_without_changing_embedding_model(
    monkeypatch,
):
    captured = []

    monkeypatch.setenv("GEMINI_API_KEY", "test-placeholder")
    monkeypatch.setenv("GEMINI_GENERATION_MODEL", "provider-verified-model")

    def fake_llm(**kwargs):
        captured.append(kwargs)
        return object()

    monkeypatch.setattr(answer_generator, "ChatGoogleGenerativeAI", fake_llm)
    generator = answer_generator.GroundedAnswerGenerator()
    assert generator.model_name == "provider-verified-model"
    assert captured[0]["model"] == "provider-verified-model"
    assert captured[0]["max_retries"] == 1
    assert captured[0]["thinking_level"] == "low"


def test_model_constructor_argument_overrides_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-placeholder")
    monkeypatch.setenv("GEMINI_GENERATION_MODEL", "configured-model")
    monkeypatch.setattr(answer_generator, "ChatGoogleGenerativeAI", lambda **kwargs: object())
    generator = answer_generator.GroundedAnswerGenerator(model_name="explicit-model")
    assert generator.model_name == "explicit-model"
