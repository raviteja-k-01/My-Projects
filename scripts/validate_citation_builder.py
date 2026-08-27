from src.generation.citation_builder import CitationBuilder


def main() -> None:
    answer = """
SEV-1 incidents must use a named Incident Commander,
and responders must prioritize safety and containment
[Evidence 2, Evidence 1].

The response actions must also preserve evidence
[Evidence 1].
""".strip()

    evidence = [
        {
            "evidence_id": 1,
            "chunk_id": "SEC-IR-004-B0021",
            "document_id": "SEC-IR-004",
            "document_name": (
                "Security Incident Response "
                "& Business Continuity Manual"
            ),
            "page_start": 13,
            "page_end": 13,
            "section_number": "3",
            "section_title": (
                "Severity Classification "
                "and Reclassification"
            ),
            "subsection_number": "3.1",
            "subsection_title": "SEV-1 Critical",
        },
        {
            "evidence_id": 2,
            "chunk_id": "SEC-IR-004-B0044",
            "document_id": "SEC-IR-004",
            "document_name": (
                "Security Incident Response "
                "& Business Continuity Manual"
            ),
            "page_start": 26,
            "page_end": 26,
            "section_number": "6",
            "section_title": (
                "Incident Command and Escalation"
            ),
            "subsection_number": "6.3",
            "subsection_title": (
                "Executive escalation"
            ),
        },
    ]

    builder = CitationBuilder()

    result = builder.build(
        answer=answer,
        evidence=evidence,
    )

    print("\nCITATION BUILDER VALIDATION")

    print("\nANSWER")
    print(result["answer"])

    print("\nCITATIONS")

    for citation in result["citations"]:
        print(
            f"\nEvidence: "
            f"{citation['evidence_id']}"
        )

        print(
            f"Document: "
            f"{citation['document_id']}"
        )

        print(
            f"Page: "
            f"{citation['page_label']}"
        )

        print(
            f"Section: "
            f"{citation['section_label']}"
        )

        print(
            f"Chunk ID: "
            f"{citation['chunk_id']}"
        )

    if result["citation_count"] != 2:
        raise ValueError(
            "Expected exactly 2 citations"
        )

    if (
        result["citations"][0]["evidence_id"]
        != 2
    ):
        raise ValueError(
            "Citation order is incorrect"
        )

    if (
        result["citations"][1]["evidence_id"]
        != 1
    ):
        raise ValueError(
            "Citation order is incorrect"
        )

    print(
        "\nCITATION BUILDER: PASS"
    )


if __name__ == "__main__":
    main()