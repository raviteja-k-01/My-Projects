import asyncio
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


def test_real_pipeline_startup_with_only_provider_boundaries_mocked(monkeypatch):
    from src.rag import rag_pipeline
    from src.api import app as api
    closed = []
    monkeypatch.setattr(rag_pipeline, "HybridRetriever", lambda: SimpleNamespace(
        retrieve_with_mode=lambda **kw: ([], "hybrid"), close=lambda: closed.append(True)))
    monkeypatch.setattr(rag_pipeline, "GroundedAnswerGenerator", lambda: SimpleNamespace(
        generate=lambda **kw: pytest.fail("Empty evidence must not generate")))
    with TestClient(api.app) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/ask", json={"question": "Unknown policy?"})
        assert response.status_code == 200
        assert "insufficient evidence" in response.json()["answer"]
    assert closed == [True]
    assert api.rag_pipeline is None


def test_lifespan_releases_resources_on_exception(monkeypatch):
    from src.api import app as api
    closed = []
    monkeypatch.setattr(api, "EnterpriseRAGPipeline", lambda: SimpleNamespace(close=lambda: closed.append(True)))
    async def run():
        with pytest.raises(RuntimeError, match="shutdown fault"):
            async with api.lifespan(api.app):
                raise RuntimeError("shutdown fault")
    asyncio.run(run())
    assert closed == [True]
    assert api.rag_pipeline is None


@pytest.mark.parametrize("name", ["compare_reranker_answer_case", "evaluate_answer_quality_ab"])
def test_legacy_commands_fail_closed_without_importing_models(name, capsys):
    module = importlib.import_module("scripts." + name)
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    assert "LEGACY EXPERIMENT" in capsys.readouterr().out


@pytest.mark.parametrize("status", [200, 503])
def test_streamlit_startup_submission_and_provider_error(monkeypatch, status):
    from streamlit.testing.v1 import AppTest
    import requests
    calls = []
    def post(url, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status_code=status, json=lambda: (
            {"answer": "Insufficient evidence.", "citations": []} if status == 200
            else {"detail": "Mock service unavailable"}))
    monkeypatch.setattr(requests, "post", post)
    app = AppTest.from_file(Path("src/ui/app.py").resolve()).run()
    assert not app.exception
    assert not calls
    app.text_area[0].set_value("Unknown policy?")
    app.button[0].click().run()
    assert not app.exception
    assert len(calls) == 1
    assert bool(app.error) == (status == 503)
