"""Bounded generation attempts and safe provider failure classifications."""
from __future__ import annotations

from dataclasses import dataclass
import re

import httpx
try:
    import httpx2
except ImportError:  # The SDK also supports environments with only httpx.
    httpx2 = None

TIMEOUT_ERRORS = (TimeoutError, httpx.TimeoutException) + ((httpx2.TimeoutException,) if httpx2 else ())
TRANSPORT_ERRORS = (httpx.TransportError,) + ((httpx2.TransportError,) if httpx2 else ())


class GenerationError(RuntimeError):
    def __init__(self, category: str, message: str, *, code: int | None = None) -> None:
        super().__init__(message)
        self.category = category
        self.code = code


@dataclass
class RequestBudget:
    """An explicitly authorized invocation budget, shared across one eval batch."""

    limit: int
    used: int = 0

    def __post_init__(self) -> None:
        if type(self.limit) is not int or self.limit < 0:
            raise ValueError("Request budget must be a nonnegative integer")

    @property
    def remaining(self) -> int:
        return self.limit - self.used

    def consume(self) -> None:
        if not self.remaining:
            raise GenerationError("budget_exhausted", "Generation request budget exhausted; no request sent.")
        self.used += 1


def classify_failure(exc: Exception) -> GenerationError:
    """Never put raw provider text (possibly containing secrets) in public errors."""
    code = getattr(exc, "code", None)
    if isinstance(exc, TIMEOUT_ERRORS):
        return GenerationError("timeout", "Gemini generation request timed out. No answer was captured.")
    if isinstance(exc, TRANSPORT_ERRORS):
        return GenerationError("transport", "Gemini generation connection failed. No answer was captured.")
    # Some LangChain versions wrap the structured SDK error. Inspect causes first.
    cause = exc.__cause__
    if code is None and cause is not None:
        return classify_failure(cause)
    if code is None:
        match = re.search(r"\b(400|401|403|404|429|500|502|503|504)\b", str(exc))
        code = int(match.group(1)) if match else None
        for name, status in (("RESOURCE_EXHAUSTED", 429), ("UNAVAILABLE", 503), ("DEADLINE_EXCEEDED", 504)):
            if name in str(exc):
                code = status
                break
    if code == 429:
        return GenerationError("quota", "Gemini generation API quota is exhausted. No answer was captured.", code=code)
    if code == 503:
        return GenerationError("unavailable", "Gemini generation model is temporarily unavailable (HTTP 503, high demand). No answer was generated.", code=code)
    if code == 504:
        return GenerationError("timeout", "Gemini generation request timed out (HTTP 504). No answer was captured.", code=code)
    if code in (500, 502):
        return GenerationError("server", f"Gemini generation service error (HTTP {code}). No answer was captured.", code=code)
    if code in (401, 403):
        return GenerationError("authentication", "Gemini generation authorization failed. No answer was captured.", code=code)
    return GenerationError("request", "Gemini generation request failed. No answer was captured.", code=code)
