import pytest

from src.generation.citation_builder import CitationBuilder


def make_evidence(
    evidence_id: int,
    chunk_id: str,
    page_start: int = 10,
    page_end: int = 10,
) -> dict:
    return {
        "evidence_id": evidence_id,
        "chunk_id": chunk_id,
        "document_id": "TEST-DOC",
        "document_name": "Test Document",
        "page_start": page_start,
        "page_end": page_end,
        "section_number": "3",
        "section_title": "Security",
        "subsection_number": "3.1",
        "subsection_title": "Access Control",
    }


def test_citation_builder_maps_single_citation():
    builder = CitationBuilder()

    evidence = [
        make_evidence(
            evidence_id=1,
            chunk_id="CHUNK-001",
        )
    ]

    result = builder.build(
        answer="MFA is required. [Evidence 1]",
        evidence=evidence,
    )

    assert result["citation_count"] == 1

    citation = result["citations"][0]

    assert citation["evidence_id"] == 1
    assert citation["chunk_id"] == "CHUNK-001"
    assert citation["document_id"] == "TEST-DOC"
    assert citation["page_label"] == "10"

    assert (
        citation["section_label"]
        == "3 | Security | 3.1 | Access Control"
    )


def test_citation_builder_supports_grouped_citations():
    builder = CitationBuilder()

    evidence = [
        make_evidence(1, "CHUNK-001"),
        make_evidence(2, "CHUNK-002"),
    ]

    result = builder.build(
        answer=(
            "The requirement is supported by multiple "
            "sources. [Evidence 1, Evidence 2]"
        ),
        evidence=evidence,
    )

    assert result["citation_count"] == 2

    assert result["citations"][0]["chunk_id"] == "CHUNK-001"
    assert result["citations"][1]["chunk_id"] == "CHUNK-002"


def test_citation_builder_removes_duplicate_citations():
    builder = CitationBuilder()

    evidence = [
        make_evidence(1, "CHUNK-001"),
        make_evidence(2, "CHUNK-002"),
    ]

    result = builder.build(
        answer=(
            "First claim. [Evidence 2] "
            "Second claim. [Evidence 1] "
            "Repeated claim. [Evidence 2]"
        ),
        evidence=evidence,
    )

    assert result["citation_count"] == 2

    assert result["citations"][0]["evidence_id"] == 2
    assert result["citations"][1]["evidence_id"] == 1


def test_citation_builder_rejects_unknown_evidence():
    builder = CitationBuilder()

    evidence = [
        make_evidence(1, "CHUNK-001"),
    ]

    with pytest.raises(
        ValueError,
        match="Answer cites Evidence 2",
    ):
        builder.build(
            answer="Unsupported citation. [Evidence 2]",
            evidence=evidence,
        )


def test_citation_builder_formats_page_range():
    builder = CitationBuilder()

    evidence = [
        make_evidence(
            evidence_id=1,
            chunk_id="CHUNK-001",
            page_start=10,
            page_end=12,
        )
    ]

    result = builder.build(
        answer="See the policy section. [Evidence 1]",
        evidence=evidence,
    )

    assert (
        result["citations"][0]["page_label"]
        == "10-12"
    )