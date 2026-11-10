from fastapi.testclient import TestClient

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
            app_module.ask(request)
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
            app_module.ask(request)
            assert False, "Expected HTTPException"

        except app_module.HTTPException as exc:
            assert exc.status_code == 503
            assert (
                exc.detail
                == "Gemini API quota is exhausted."
            )

    finally:
        app_module.rag_pipeline = original_pipeline