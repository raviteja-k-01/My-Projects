import pytest

from src.retrieval.context_builder import ContextBuilder


def make_candidate(
    chunk_id: str,
    text: str,
    token_count: int = 10,
    rank: int = 1,
) -> dict:
    return {
        "rank": rank,
        "chunk": {
            "chunk_id": chunk_id,
            "document_id": "TEST-DOC",
            "document_name": "Test Document",
            "page_start": 5,
            "page_end": 5,
            "section_number": "3",
            "section_title": "Security",
            "subsection_number": "3.1",
            "subsection_title": "Access Control",
            "content_type": "body",
            "text": text,
            "token_count": token_count,
        },
    }


def test_context_builder_creates_evidence_packet():
    builder = ContextBuilder()

    candidates = [
        make_candidate(
            chunk_id="CHUNK-001",
            text="Employees must use multi-factor authentication.",
        ),
        make_candidate(
            chunk_id="CHUNK-002",
            text="Passwords must never be shared with other users.",
            rank=2,
        ),
    ]

    result = builder.build(candidates)

    assert result["evidence_count"] == 2
    assert result["total_tokens"] == 20

    assert result["evidence"][0]["evidence_id"] == 1
    assert result["evidence"][0]["chunk_id"] == "CHUNK-001"

    assert result["evidence"][1]["evidence_id"] == 2
    assert result["evidence"][1]["chunk_id"] == "CHUNK-002"

    assert "[Evidence 1]" in result["context"]
    assert "[Evidence 2]" in result["context"]

    assert "Document: TEST-DOC" in result["context"]
    assert "Page: 5" in result["context"]

    assert (
        "Section: 3 | Security | 3.1 | Access Control"
        in result["context"]
    )


def test_context_builder_removes_near_duplicates():
    builder = ContextBuilder()

    same_text = (
        "Employees must use approved company devices "
        "when accessing enterprise systems."
    )

    candidates = [
        make_candidate(
            chunk_id="CHUNK-001",
            text=same_text,
        ),
        make_candidate(
            chunk_id="CHUNK-002",
            text=same_text,
            rank=2,
        ),
    ]

    result = builder.build(candidates)

    assert result["evidence_count"] == 1
    assert result["evidence"][0]["chunk_id"] == "CHUNK-001"


def test_context_builder_respects_token_budget():
    builder = ContextBuilder(
        max_context_tokens=15,
    )

    candidates = [
        make_candidate(
            chunk_id="CHUNK-001",
            text="First evidence chunk.",
            token_count=10,
        ),
        make_candidate(
            chunk_id="CHUNK-002",
            text="Second evidence chunk.",
            token_count=10,
            rank=2,
        ),
    ]

    result = builder.build(candidates)

    assert result["evidence_count"] == 1
    assert result["total_tokens"] == 10
    assert result["evidence"][0]["chunk_id"] == "CHUNK-001"


def test_context_builder_rejects_missing_chunk_id():
    builder = ContextBuilder()

    candidates = [
        {
            "rank": 1,
            "chunk": {
                "text": "Evidence without a chunk ID.",
                "token_count": 10,
            },
        }
    ]

    with pytest.raises(
        ValueError,
        match="Candidate chunk has no chunk_id",
    ):
        builder.build(candidates)