"""Validate local demo backend configuration and BM25 assets without network calls.

Run from the repository root:
    python -m scripts.deployment_preflight

This checks only local configuration and the sparse index. It does NOT certify
Gemini generation, Qdrant connectivity, public hosting, or answer quality.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from dotenv import load_dotenv

ALLOWED_EMBEDDING_KEYS = frozenset({"GEMINI_API_KEY", "GEMINI_API_KEY_1"})
REQUIRED_NAMES = ("QDRANT_URL", "QDRANT_API_KEY")


class PreflightError(ValueError):
    """Expected local configuration or index validation failure."""


def validate_config(environ: Mapping[str, str]) -> None:
    """Validate required ENV names without exposing secret contents."""
    public_auth_required = environ.get("RAG_REQUIRE_DEMO_AUTH", "").strip().lower() in {"1", "true", "yes"}
    if public_auth_required:
        if environ.get("RAG_DEMO_MODE", "").strip().lower() not in {"1", "true", "yes"}:
            raise PreflightError("Public deployment requires RAG_DEMO_MODE")
        if len(environ.get("RAG_DEMO_API_TOKEN", "")) < 24:
            raise PreflightError("Public deployment requires RAG_DEMO_API_TOKEN with at least 24 characters")

    provider = environ.get("RAG_GENERATION_PROVIDER", "gemini").strip().lower()
    if provider not in {"gemini", "groq"}:
        raise PreflightError("RAG_GENERATION_PROVIDER must be gemini or groq")

    selected = environ.get("GEMINI_EMBEDDING_KEY_ENV", "GEMINI_API_KEY").strip()
    if selected not in ALLOWED_EMBEDDING_KEYS:
        raise PreflightError("GEMINI_EMBEDDING_KEY_ENV must name an allowed key variable")

    required = REQUIRED_NAMES
    if selected == "GEMINI_API_KEY":
        required += ("GEMINI_API_KEY",)
    if provider == "groq":
        required += ("GROQ_API_KEY",)
    else:
        generation_key = environ.get("GEMINI_GENERATION_KEY_ENV", "GEMINI_API_KEY").strip()
        if generation_key not in ALLOWED_EMBEDDING_KEYS:
            raise PreflightError("GEMINI_GENERATION_KEY_ENV must name an allowed key variable")
        if generation_key == "GEMINI_API_KEY":
            required += ("GEMINI_API_KEY",)

    if environ.get("RAG_DEMO_MODE", "").strip().lower() in {"1", "true", "yes"}:
        required += ("RAG_DEMO_API_TOKEN",)
        if len(environ.get("RAG_DEMO_API_TOKEN", "")) < 24:
            raise PreflightError("RAG_DEMO_API_TOKEN must have at least 24 characters")
        if int(environ.get("RAG_RATE_LIMIT_PER_MINUTE", "12")) > 3:
            raise PreflightError("Public demo must limit requests per minute to at most 3")
    missing = [key for key in dict.fromkeys(required) if not environ.get(key, "").strip()]
    if missing:
        raise PreflightError("Missing required environment variables: " + ", ".join(missing))
    if not environ.get(selected, "").strip():
        raise PreflightError(f"Selected embedding environment variable is empty: {selected}")

    if provider == "gemini":
        generation_key = environ.get("GEMINI_GENERATION_KEY_ENV", "GEMINI_API_KEY").strip()
        if not environ.get(generation_key, "").strip():
            raise PreflightError(f"Selected generation environment variable is empty: {generation_key}")

    if not environ["QDRANT_URL"].strip().startswith(("http://", "https://")):
        raise PreflightError("QDRANT_URL must be an HTTP(S) URL")


def validate_bm25(
    path: Path,
    *,
    expected_count: int | None = None,
    index_factory: Callable[[Path], Any] | None = None,
) -> int:
    """Load the actual sparse index. Local files only; never calls an API."""
    if not path.is_dir():
        raise PreflightError("BM25 directory is missing; mount the existing data/bm25 directory")
    if expected_count is not None and expected_count <= 0:
        raise PreflightError("Expected BM25 chunk count must be positive")
    if index_factory is None:
        from src.indexing.bm25_index import BM25Index
        index_factory = lambda p: BM25Index(index_path=p)
    try:
        index = index_factory(path)
        index.load()
        corpus = index.retriever.corpus
        count = len(corpus)
    except Exception as exc:
        # Avoid printing exception details which could include local path or data.
        raise PreflightError("BM25 index is not loadable: verify compatible artifacts") from exc
    if count == 0:
        raise PreflightError("BM25 index is empty")
    if expected_count is not None and count != expected_count:
        raise PreflightError(
            f"BM25 chunk count mismatch: expected {expected_count}, found {count}"
        )
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bm25-dir", type=Path, default=Path("data/bm25"))
    parser.add_argument("--expect-chunks", type=int, default=None)
    args = parser.parse_args()
    load_dotenv()
    try:
        validate_config(os.environ)
        count = validate_bm25(args.bm25_dir, expected_count=args.expect_chunks)
    except PreflightError as exc:
        parser.exit(status=2, message=f"DEPLOY_PREFLIGHT_FAILED: {exc}\n")
    print(f"DEPLOY_PREFLIGHT_OK: BM25 {count} chunks; env names present; network calls 0")
    print("External service availability and response quality: NOT VERIFIED")


if __name__ == "__main__":
    main()
