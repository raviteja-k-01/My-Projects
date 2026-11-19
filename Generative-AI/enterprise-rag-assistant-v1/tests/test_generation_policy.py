from types import SimpleNamespace

import httpx
import pytest
from langchain_core.runnables import RunnableLambda

from src.generation import answer_generator as module
from src.generation.reliability import GenerationError, RequestBudget


def generator(monkeypatch, outcomes, *, attempts=3, budget=3):
    monkeypatch.setenv("GEMINI_API_KEY", "offline-placeholder")
    options, calls, waits, events = [], [], [], []
    def fake(**kwargs):
        options.append(kwargs)
        return RunnableLambda(lambda value: value)
    monkeypatch.setattr(module, "ChatGoogleGenerativeAI", fake)
    monkeypatch.setattr(module.time, "sleep", waits.append)
    obj = module.GroundedAnswerGenerator(max_attempts=attempts)
    def invoke(value):
        calls.append(value)
        result = outcomes.pop(0)
        if isinstance(result, Exception):
            raise result
        return result
    obj.chain = SimpleNamespace(invoke=invoke)
    obj.request_budget = RequestBudget(budget)
    obj.observer = events.append
    return obj, calls, waits, events, options


def success(**kwargs):
    return SimpleNamespace(text="Supported [Evidence 1]", usage_metadata={}, response_metadata={"finish_reason": "STOP", **kwargs})


def test_transient_backoff_and_single_sdk_attempt(monkeypatch):
    obj, calls, waits, events, options = generator(monkeypatch, [
        module.errors.ServerError(503, {}), httpx.ReadTimeout("secret-url"), success()])
    assert obj.generate("Question", "Source")["answer"] == "Supported [Evidence 1]"
    assert len(calls) == 3 and obj.request_budget.used == 3
    assert 1 <= waits[0] <= 1.5 and 2 <= waits[1] <= 2.5
    assert options[0]["max_retries"] == 1
    assert [e.get("category") for e in events if e["event"] == "failed"] == ["unavailable", "timeout"]
    assert "secret-url" not in str(events)


@pytest.mark.parametrize("code,category", [(429,"quota"),(401,"authentication"),(403,"authentication"),(400,"request")])
def test_quota_and_auth_never_retry_or_rotate(monkeypatch, code, category):
    obj, calls, waits, events, _ = generator(monkeypatch, [module.errors.ClientError(code, {"error":{"message":"secret-key"}})])
    with pytest.raises(GenerationError) as caught:
        obj.generate("Question", "Source")
    assert caught.value.category == category
    assert len(calls) == 1 and not waits
    assert "secret-key" not in str(caught.value)
    assert obj.key_env == "GEMINI_API_KEY"


def test_budget_includes_retry_attempts(monkeypatch):
    obj, calls, waits, _, _ = generator(monkeypatch, [module.errors.ServerError(503, {})], budget=1)
    with pytest.raises(GenerationError, match="503"):
        obj.generate("Question", "Source")
    with pytest.raises(GenerationError, match="budget exhausted"):
        obj.generate("Next", "Source")
    assert len(calls) == 1 and not waits


def test_sdk_httpx2_transport_is_also_classified(monkeypatch):
    import httpx2
    obj,calls,waits,events,_ = generator(monkeypatch,[httpx2.ReadTimeout('secret'),success()])
    obj.generate('Question','Source')
    assert len(calls) == 2 and len(waits) == 1
    assert events[1]['category'] == 'timeout'


@pytest.mark.parametrize("finish", ["MAX_TOKENS", "SAFETY", "RECITATION"])
def test_partial_or_blocked_output_is_not_saved_as_answer(monkeypatch, finish):
    obj, calls, waits, _, _ = generator(monkeypatch, [success(finish_reason=finish)])
    with pytest.raises(GenerationError, match="complete answer"):
        obj.generate("Question", "Source")
    assert len(calls) == 1 and not waits


@pytest.mark.parametrize("response_metadata", [{}, {"finish_reason": None}, {"finish_reason": ""}])
def test_missing_finish_reason_fails_closed(monkeypatch, response_metadata):
    """A successful HTTP response is not proof of a complete model answer."""
    response = SimpleNamespace(
        text="Looks complete [Evidence 1]",
        usage_metadata={},
        response_metadata=response_metadata,
    )
    obj, calls, waits, _, _ = generator(monkeypatch, [response])
    with pytest.raises(GenerationError, match="complete answer"):
        obj.generate("Question", "Source")
    assert len(calls) == 1 and not waits


@pytest.mark.parametrize("model_text", [None, [], "", "   "])
def test_missing_or_empty_text_is_rejected(monkeypatch, model_text):
    response = SimpleNamespace(
        text=model_text, usage_metadata={}, response_metadata={"finish_reason": "STOP"},
    )
    obj, calls, waits, _, _ = generator(monkeypatch, [response])
    with pytest.raises(GenerationError, match="empty answer"):
        obj.generate("Question", "Source")
    assert len(calls) == 1 and not waits


def test_explicit_generation_credential_selection(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY_1", "alternate-placeholder")
    monkeypatch.setenv("GEMINI_GENERATION_KEY_ENV", "GEMINI_API_KEY_1")
    obj, _, _, _, opts = generator(monkeypatch, [])
    assert opts[0]["api_key"] == "alternate-placeholder"
    assert obj.configuration["key_env"] == "GEMINI_API_KEY_1"
    assert "alternate-placeholder" not in str(obj.configuration)


@pytest.mark.parametrize("kwargs", [{"max_attempts":0},{"max_attempts":6},{"timeout_seconds":float("nan")},{"timeout_seconds":121}])
def test_invalid_generation_limits_fail_before_client(monkeypatch, kwargs):
    monkeypatch.setenv("GEMINI_API_KEY", "placeholder")
    monkeypatch.setattr(module, "ChatGoogleGenerativeAI", lambda **kw: pytest.fail("Invalid config must not create client"))
    with pytest.raises(ValueError):
        module.GroundedAnswerGenerator(**kwargs)
