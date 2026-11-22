"""Public-demo safeguards and BM25 artifact loading; strictly offline."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import hashlib
import os
from zipfile import ZipFile, ZIP_DEFLATED

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from scripts.cloud_bm25_install import install_archive, download_archive
from scripts.package_bm25_for_demo import package
from scripts.deployment_preflight import validate_config, PreflightError
from src.api import app as api


def test_bm25_archive_package_install_without_reindexing(tmp_path):
    original = tmp_path / "local" / "bm25"
    original.mkdir(parents=True)
    (original / "index.npz").write_bytes(b"correct-index-content")
    (original / "vocab.json").write_text('{"demo":"policy"}')
    out = tmp_path / "release" / "bm25_demo.zip"
    result = package(original, out)
    assert result["files"] == 2
    dest = tmp_path / "other" / "bm25"
    assert install_archive(out.read_bytes(), expected_sha256=result["sha256"], destination=dest) == 2
    assert (dest / "index.npz").read_bytes() == b"correct-index-content"
    assert (dest / "vocab.json").read_text() == '{"demo":"policy"}'
    assert (original / "index.npz").read_bytes() == b"correct-index-content"
    with pytest.raises(ValueError, match="exists"):
        install_archive(out.read_bytes(), expected_sha256=result["sha256"], destination=dest)


@pytest.mark.parametrize("unsafe", ["../steal.txt", "/abs/bm25", "bm25/../steal.txt",
                                      "bm25/back\\slash.json"])
def test_bm25_zip_path_traversal_is_rejected(tmp_path, unsafe):
    data = BytesIO()
    with ZipFile(data, "w", compression=ZIP_DEFLATED) as z:
        z.writestr(unsafe, b"not permitted")
    content = data.getvalue()
    target = tmp_path / "bm25"
    with pytest.raises(ValueError, match="Unsafe"):
        install_archive(content, expected_sha256=hashlib.sha256(content).hexdigest(),
                        destination=target)
    assert not target.exists()


def test_bm25_integrity_mismatch_is_rejected(tmp_path):
    data = b"not original"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        install_archive(data, expected_sha256="a" * 64, destination=tmp_path / "bm25")


def test_archive_download_rejects_insecure_urls():
    with pytest.raises(ValueError, match="HTTPS"):
        download_archive("http://example.com/archive.zip")
    with pytest.raises(ValueError, match="credentials"):
        download_archive("https://username:password@example.com/archive.zip")


def request(token=""):
    headers = [(b"x-rag-demo-token", token.encode())] if token else []
    return Request({
        "type": "http", "method": "POST", "path": "/ask",
        "headers": headers, "client": ("127.0.0.1", 12345),
    })


def test_demo_api_token_blocks_before_model_queries(monkeypatch):
    monkeypatch.setenv("RAG_DEMO_MODE", "1")
    monkeypatch.setenv("RAG_DEMO_API_TOKEN", "valid-demo-token-of-at-least-24-chars")
    with pytest.raises(HTTPException) as exc:
        api.enforce_rate_limit(request())
    assert exc.value.status_code == 401
    with pytest.raises(HTTPException) as exc:
        api.enforce_rate_limit(request("incorrect-token"))
    assert exc.value.status_code == 401


def test_demo_global_hourly_guard(monkeypatch):
    monkeypatch.setenv("RAG_DEMO_MODE", "1")
    monkeypatch.setenv("RAG_DEMO_API_TOKEN", "valid-demo-token-of-at-least-24-chars")
    monkeypatch.setattr(api.demo_daily_limiter, "allow", lambda client_id: True)
    monkeypatch.setattr(api.demo_hourly_limiter, "allow", lambda client_id: False)
    with pytest.raises(HTTPException) as exc:
        api.enforce_rate_limit(request("valid-demo-token-of-at-least-24-chars"))
    assert exc.value.status_code == 429


def test_public_preflight_rejects_missing_or_weak_token():
    env = {"RAG_DEMO_MODE": "1", "RAG_GENERATION_PROVIDER": "groq",
           "RAG_RATE_LIMIT_PER_MINUTE": "1",
           "GROQ_API_KEY": "secret", "GEMINI_API_KEY": "secret",
           "QDRANT_URL": "https://demo.example.com", "QDRANT_API_KEY": "secret"}
    with pytest.raises(PreflightError, match="at least 24"):
        validate_config(env)
    env["RAG_DEMO_API_TOKEN"] = "valid-demo-token-of-at-least-24-chars"
    validate_config(env)
    env["RAG_RATE_LIMIT_PER_MINUTE"] = "12"
    with pytest.raises(PreflightError, match="at most 3"):
        validate_config(env)
