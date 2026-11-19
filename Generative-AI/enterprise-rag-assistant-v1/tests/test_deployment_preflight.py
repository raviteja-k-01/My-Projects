"""No-network preflight regression tests. No credentials or external services."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.deployment_preflight import PreflightError, validate_bm25, validate_config


VALID_ENV = {
    "GEMINI_API_KEY": "local-test-only",
    "QDRANT_URL": "https://example.invalid",
    "QDRANT_API_KEY": "local-test-only",
}


def test_required_configuration() -> None:
    validate_config(VALID_ENV)


@pytest.mark.parametrize("missing", ["GEMINI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"])
def test_missing_var_rejected(missing: str) -> None:
    with pytest.raises(PreflightError, match="Missing required"):
        validate_config({k: v for k, v in VALID_ENV.items() if k != missing})


def test_selected_second_embedding_key_requires_value() -> None:
    with pytest.raises(PreflightError, match="Selected embedding"):
        validate_config({**VALID_ENV, "GEMINI_EMBEDDING_KEY_ENV": "GEMINI_API_KEY_1"})


def test_allowed_second_embedding_key() -> None:
    validate_config({**VALID_ENV, "GEMINI_EMBEDDING_KEY_ENV": "GEMINI_API_KEY_1", "GEMINI_API_KEY_1": "local-test-only"})


def test_both_services_can_explicitly_select_secondary_without_primary():
    values = {k:v for k,v in VALID_ENV.items() if k != "GEMINI_API_KEY"}
    validate_config({**values,"GEMINI_API_KEY_1":"secondary-test-only",
                     "GEMINI_EMBEDDING_KEY_ENV":"GEMINI_API_KEY_1",
                     "GEMINI_GENERATION_KEY_ENV":"GEMINI_API_KEY_1"})


def test_selected_generation_key_must_exist():
    with pytest.raises(PreflightError,match="Selected generation"):
        validate_config({**VALID_ENV,"GEMINI_GENERATION_KEY_ENV":"GEMINI_API_KEY_1"})


def test_invalid_key_selector_rejected() -> None:
    with pytest.raises(PreflightError, match="allowed"):
        validate_config({**VALID_ENV, "GEMINI_EMBEDDING_KEY_ENV": "OTHER_KEY"})


def test_invalid_url_rejected() -> None:
    with pytest.raises(PreflightError, match="HTTP"):
        validate_config({**VALID_ENV, "QDRANT_URL": "not-a-url"})


def test_missing_index_dir_rejected(tmp_path: Path) -> None:
    with pytest.raises(PreflightError, match="BM25 directory"):
        validate_bm25(tmp_path / "missing")


def test_index_load_and_count(tmp_path: Path) -> None:
    loaded = []
    class StubIndex:
        retriever = SimpleNamespace(corpus=[{}, {}, {}])
        def load(self):
            loaded.append(True)
    assert validate_bm25(tmp_path, expected_count=3, index_factory=lambda p: StubIndex()) == 3
    assert loaded == [True]


def test_index_count_mismatch_rejected(tmp_path: Path) -> None:
    class StubIndex:
        retriever = SimpleNamespace(corpus=[{}, {}])
        def load(self):
            pass
    with pytest.raises(PreflightError, match="chunk count mismatch"):
        validate_bm25(tmp_path, expected_count=3, index_factory=lambda p: StubIndex())


def test_broken_index_rejected(tmp_path: Path) -> None:
    class StubIndex:
        def load(self):
            raise FileNotFoundError("fake")
    with pytest.raises(PreflightError, match="not loadable"):
        validate_bm25(tmp_path, index_factory=lambda p: StubIndex())


def test_empty_index_rejected(tmp_path: Path) -> None:
    class StubIndex:
        retriever = SimpleNamespace(corpus=[])
        def load(self):
            pass
    with pytest.raises(PreflightError, match="empty"):
        validate_bm25(tmp_path, index_factory=lambda p: StubIndex())


def test_non_positive_expectation_rejected(tmp_path: Path) -> None:
    with pytest.raises(PreflightError, match="positive"):
        validate_bm25(tmp_path, expected_count=0, index_factory=lambda p: None)
