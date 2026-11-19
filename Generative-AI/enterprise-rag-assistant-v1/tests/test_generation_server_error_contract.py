"""Offline contract tests for transient Gemini generation failures.

No API credentials, network calls, or production-code changes required.
"""

import pytest
from google.genai import errors as genai_errors

from src.generation.answer_generator import GroundedAnswerGenerator


@pytest.mark.parametrize(
    ("status_code", "message_fragment"),
    [(503, "temporarily unavailable"), (504, "timed out")],
)
def test_server_error_is_reported_as_generation_failure(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    message_fragment: str,
) -> None:
    class FakeServerError(Exception):
        def __init__(self, code: int) -> None:
            super().__init__(f"simulated HTTP {code}")
            self.code = code

    class FailingChain:
        def invoke(self, _payload: object) -> None:
            raise FakeServerError(status_code)

    # Replace only the error type; do not construct a Gemini client.
    monkeypatch.setattr(genai_errors, "ServerError", FakeServerError)
    generator = object.__new__(GroundedAnswerGenerator)
    generator.chain = FailingChain()

    with pytest.raises(RuntimeError, match=message_fragment):
        generator.generate(question="offline test", context="[Evidence 1] Offline evidence")
