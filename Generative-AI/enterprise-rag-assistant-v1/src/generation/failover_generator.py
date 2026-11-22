"""Opt-in live generation fallback: Gemini -> Groq GPT-OSS.

No retrieval, embedding, index, or benchmark generation path is changed.
"""
from __future__ import annotations

import logging
import math
import os
import re
from typing import Any

import requests

from src.generation.reliability import GenerationError
from src.generation.citation_normalizer import normalize_explicit_citations

LOGGER = logging.getLogger(__name__)
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
TRANSIENT_PRIMARY_CATEGORIES = frozenset(
    {"unavailable", "timeout", "server", "transport", "quota"}
)


def _retry_after_seconds(response: requests.Response) -> float | None:
    value = response.headers.get("retry-after")
    if not value:
        return None
    value = value.strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", value):
        seconds = float(value)
        return seconds if 0 <= seconds <= 86400 else None
    return None


def _header_value(headers: requests.structures.CaseInsensitiveDict, *names: str) -> str | None:
    for name in names:
        value = headers.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


class GroqAnswerGenerator:
    """One bounded Groq request using the existing grounding instructions."""

    def __init__(self, *, system_prompt: str) -> None:
        self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError("GROQ_FALLBACK_ENABLED requires GROQ_API_KEY")
        self.model_name = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-120b").strip()
        self.timeout_seconds = float(os.getenv("GROQ_FALLBACK_TIMEOUT_SECONDS", "35"))
        self.max_output_tokens = int(os.getenv("GROQ_FALLBACK_MAX_OUTPUT_TOKENS", "2048"))
        if not self.model_name:
            raise ValueError("GROQ_FALLBACK_MODEL cannot be empty")
        if not math.isfinite(self.timeout_seconds) or not 1 <= self.timeout_seconds <= 120:
            raise ValueError("GROQ_FALLBACK_TIMEOUT_SECONDS must be between 1 and 120")
        if not 1 <= self.max_output_tokens <= 8192:
            raise ValueError("GROQ_FALLBACK_MAX_OUTPUT_TOKENS must be between 1 and 8192")
        self.system_prompt = system_prompt

    def generate(self, question: str, context: str) -> dict[str, Any]:
        question, context = question.strip(), context.strip()
        if not question or not context:
            raise ValueError("Question and evidence context cannot be empty")
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": (
                    f"Question:\n{question}\n\nEvidence:\n{context}\n\n"
                    "Answer the question using only the evidence above. "
                    "Cite each supported factual claim using [Evidence N] (for example, "
                    "[Evidence 1] or [Evidence 1, Evidence 2]). Repeat the word "
                    "Evidence before every number; do not use bare numbers or citation ranges."
                )},
            ],
            "max_completion_tokens": self.max_output_tokens,
            "reasoning_effort": "low",
            "temperature": 0.1,
        }
        try:
            response = requests.post(
                GROQ_CHAT_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=(5, self.timeout_seconds),
            )
        except requests.Timeout as exc:
            raise GenerationError("timeout", "Groq fallback timed out. No answer was generated.") from exc
        except requests.RequestException as exc:
            raise GenerationError("transport", "Groq fallback connection failed. No answer was generated.") from exc

        if response.status_code != 200:
            code = response.status_code
            category = (
                "quota" if code == 429 else
                "authentication" if code in (401, 403) else
                "unavailable" if code in (500, 502, 503, 504) else
                "request"
            )
            headers = response.headers
            raise GenerationError(
                category,
                f"Groq fallback failed (HTTP {code}). No answer was generated.",
                code=code,
                retry_after_seconds=_retry_after_seconds(response),
                ratelimit_remaining_requests=_header_value(headers, "x-ratelimit-remaining-requests"),
                ratelimit_remaining_tokens=_header_value(headers, "x-ratelimit-remaining-tokens"),
                ratelimit_reset_requests=_header_value(headers, "x-ratelimit-reset-requests"),
                ratelimit_reset_tokens=_header_value(headers, "x-ratelimit-reset-tokens"),
            )
        try:
            data = response.json()
            choice = data["choices"][0]
            answer = choice["message"]["content"]
            finish = choice["finish_reason"]
            usage = data.get("usage") or {}
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            raise GenerationError("invalid_response", "Groq fallback returned an invalid response.") from exc
        if finish != "stop" or not isinstance(answer, str) or not answer.strip():
            raise GenerationError("invalid_response", "Groq fallback did not return a complete answer.")
        # Groq sometimes uses explicit but noncanonical citation punctuation.
        # Repair notation deterministically; CitationBuilder still rejects
        # malformed or nonexistent references. No new IDs are invented.
        return {"answer": normalize_explicit_citations(answer.strip()),
                "model": self.model_name, "usage": usage}


class FailoverAnswerGenerator:
    """Fail over once, only on recognized primary service failures."""

    def __init__(self, primary: Any, fallback: GroqAnswerGenerator) -> None:
        self.primary = primary
        self.fallback = fallback

    def generate(self, question: str, context: str) -> dict[str, Any]:
        try:
            result = self.primary.generate(question, context)
            return result
        except GenerationError as exc:
            if exc.category not in TRANSIENT_PRIMARY_CATEGORIES:
                raise
            # Never log prompts, credentials, or raw provider error bodies.
            LOGGER.warning(
                "Primary generation failed: category=%s code=%s; attempting Groq fallback",
                exc.category, exc.code,
            )
            result = self.fallback.generate(question, context)
            LOGGER.info("Groq fallback succeeded: model=%s", result["model"])
            return {**result, "generation_fallback_used": True}
