from fastapi.testclient import TestClient
import pytest
from fastapi import Response
import src.api.app as app_module


class FakePipeline:
    def answer(self, question: str) -> dict:
        return {
            "answer": f"Test answer for: {question}",
            "citations": [],
        }

    def close(self) -> None:
        pass


def test_health_endpoint(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "EnterpriseRAGPipeline",
        FakePipeline,
    )

    with TestClient(app_module.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok"
    }


def test_livez_and_readyz_distinguish_process_and_pipeline(monkeypatch):
    monkeypatch.setattr(app_module, "EnterpriseRAGPipeline", FakePipeline)
    with TestClient(app_module.app) as client:
        live_response = client.get("/livez")
        ready_response = client.get("/readyz")
    assert live_response.status_code == 200
    assert live_response.json() == {"status": "ok"}
    assert ready_response.status_code == 503


def test_public_startup_requires_auth(monkeypatch):
    monkeypatch.setenv("RAG_REQUIRE_DEMO_AUTH", "1")
    monkeypatch.delenv("RAG_DEMO_MODE", raising=False)
    monkeypatch.delenv("RAG_DEMO_API_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="requires demo authentication"):
        with TestClient(app_module.app):
            pass


def test_ask_endpoint_returns_answer(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "EnterpriseRAGPipeline",
        FakePipeline,
    )

    with TestClient(app_module.app) as client:
        response = client.post(
            "/ask",
            json={
                "question": "What is the policy?"
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["answer"]
        == "Test answer for: What is the policy?"
    )

    assert data["citations"] == []


def test_evaluation_trace_is_opt_in(monkeypatch):
    class TracePipeline:
        def answer(self, question: str, evaluation_trace: bool = False):
            assert evaluation_trace is True
            return {"answer": "trace-test", "citations": [], "evaluation_trace": {
                "retrieved_chunk_ids": ["C001"],
                "selected_evidence": [],
            }}

        def close(self):
            pass

    monkeypatch.setattr(app_module, "EnterpriseRAGPipeline", TracePipeline)
    monkeypatch.setenv("RAG_ALLOW_EVALUATION_TRACE", "1")
    with TestClient(app_module.app) as client:
        enabled = client.post(
            "/ask",
            json={"question": "What is policy?", "evaluation_trace": True},
        )
    assert enabled.status_code == 200
    assert enabled.json()["evaluation_trace"]["retrieved_chunk_ids"] == ["C001"]

    monkeypatch.delenv("RAG_ALLOW_EVALUATION_TRACE", raising=False)
    with TestClient(app_module.app) as client:
        disabled = client.post(
            "/ask",
            json={"question": "What is policy?", "evaluation_trace": True},
        )
    assert disabled.status_code == 403
    assert disabled.json()["detail"] == "Evaluation trace is disabled."


def test_ask_rejects_missing_question(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "EnterpriseRAGPipeline",
        FakePipeline,
    )

    with TestClient(app_module.app) as client:
        response = client.post(
            "/ask",
            json={},
        )

    assert response.status_code == 422


def test_ask_returns_503_when_pipeline_not_ready():
    original_pipeline = app_module.rag_pipeline

    try:
        app_module.rag_pipeline = None

        request = app_module.AskRequest(
            question="Test question"
        )

        try:
            app_module.ask(request,Response())
            assert False, "Expected HTTPException"

        except app_module.HTTPException as exc:
            assert exc.status_code == 503
            assert (
                exc.detail
                == "RAG pipeline is not ready."
            )

    finally:
        app_module.rag_pipeline = original_pipeline


def test_ask_converts_runtime_error_to_503():
    class FailingPipeline:
        def answer(self, question: str):
            raise RuntimeError(
                "Gemini API quota is exhausted."
            )

    original_pipeline = app_module.rag_pipeline

    try:
        app_module.rag_pipeline = FailingPipeline()

        request = app_module.AskRequest(
            question="Test question"
        )

        try:
            app_module.ask(request,Response())
            assert False, "Expected HTTPException"

        except app_module.HTTPException as exc:
            assert exc.status_code == 503
            assert (
                exc.detail
                == "RAG service is temporarily unavailable. Please retry later."
            )

    finally:
        app_module.rag_pipeline = original_pipeline


def test_validation_error_is_classified_without_leaking_exception(monkeypatch, caplog):
    """A failed answer keeps raw content out of public errors and logs."""
    class FailingPipeline:
        def answer(self, question: str):
            raise ValueError(
                "Answer cites Evidence 999, but that evidence does not exist"
            )

        def close(self):
            pass

    monkeypatch.setattr(app_module, "EnterpriseRAGPipeline", FailingPipeline)
    with TestClient(app_module.app) as client:
        with caplog.at_level("WARNING", logger="uvicorn.error"):
            response = client.post(
                "/ask", json={"question": "Secret policy question"}
            )
    assert response.status_code == 502
    assert response.json()["detail"] == (
        "The RAG service could not produce a valid grounded answer."
    )
    assert "RAG_VALIDATION_FAILED category=citation_unknown_evidence" in caplog.text
    assert "Evidence 999" not in caplog.text
    assert "Secret policy question" not in caplog.text


def test_validation_logging_categories_are_sanitized():
    cases = {
        "Malformed or unbracketed evidence reference": "citation_malformed",
        "Duplicate evidence IDs": "citation_duplicate_ids",
        "Answer cannot be empty": "answer_empty",
        "Evidence cannot be empty": "evidence_empty",
        "Secret document text": "other_validation_error",
    }
    for message, category in cases.items():
        assert app_module.classify_validation_error(ValueError(message)) == category
