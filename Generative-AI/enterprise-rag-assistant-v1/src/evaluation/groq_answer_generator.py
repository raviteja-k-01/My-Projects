"""Groq generation adapter for source-locked, rate-bounded answer evaluation.

Live /ask uses GroqAnswerGenerator directly. This adapter ONLY adds the
benchmark's provenance, attempt budget and safe event ledger hooks.
"""
from __future__ import annotations

import hashlib
from importlib.metadata import version
from pathlib import Path
import time
from typing import Any

from dotenv import load_dotenv

from src.generation.answer_generator import SYSTEM_PROMPT
from src.generation.failover_generator import GroqAnswerGenerator
from src.generation.reliability import RequestBudget


class EvaluatedGroqAnswerGenerator:
    """One Groq request per generate call, with existing replay budget hooks."""

    def __init__(self, model_name: str | None = None) -> None:
        load_dotenv()
        self.provider = GroqAnswerGenerator(system_prompt=SYSTEM_PROMPT)
        self.model_name = self.provider.model_name
        if model_name is not None and model_name.strip() != self.model_name:
            raise ValueError(
                "For Groq evaluation, configure GROQ_FALLBACK_MODEL in .env "
                "or omit --model; no model change was made."
            )
        self.request_budget: RequestBudget | None = None
        self.observer = None

        src = Path(__file__).resolve().parents[1]
        relative_sources = (
            "generation/failover_generator.py",
            "generation/citation_normalizer.py",
            "evaluation/groq_answer_generator.py",
        )
        # Distinguish Groq from Gemini even when both share the same model name
        # or the historical Gemini code hashes stay unchanged.
        self.configuration: dict[str, Any] = {
            "provider": "groq",
            "model": self.model_name,
            "max_attempts": 1,
            "max_output_tokens": self.provider.max_output_tokens,
            "timeout_seconds": self.provider.timeout_seconds,
            "reasoning_effort": "low",
            "temperature": 0.1,
            "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            "source_sha256": {
                name: hashlib.sha256((src / name).read_bytes()).hexdigest()
                for name in relative_sources
            },
            "requests_version": version("requests"),
        }

    def generate(self, question: str, context: str) -> dict[str, Any]:
        if self.request_budget is not None:
            self.request_budget.consume()
        if self.observer:
            self.observer({"event": "started", "attempt": 1})
        start = time.perf_counter()
        try:
            result = self.provider.generate(question, context)
        except Exception as exc:
            if self.observer:
                self.observer({
                    "event": "failed", "attempt": 1,
                    "category": getattr(exc, "category", "generation_error"),
                    "seconds": time.perf_counter() - start,
                })
            raise
        if self.observer:
            self.observer({
                "event": "succeeded", "attempt": 1,
                "seconds": time.perf_counter() - start,
            })
        return result
