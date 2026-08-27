from typing import Any


def evaluate_citations(
    cited_chunk_ids: list[str],
    evidence_chunk_ids: list[str],
    expected_chunk_ids: list[str],
) -> dict[str, Any]:
    """
    Evaluate citations produced by one RAG answer.

    Metrics:
    - citation validity:
      Were all citations drawn from the supplied evidence?

    - gold citation recall:
      How many expected gold chunks were actually cited?

    Important:
    Gold citation recall can underestimate quality because
    another genuinely relevant chunk may support the answer.
    """

    cited = [
        chunk_id.strip()
        for chunk_id in cited_chunk_ids
        if isinstance(chunk_id, str)
        and chunk_id.strip()
    ]

    evidence = {
        chunk_id.strip()
        for chunk_id in evidence_chunk_ids
        if isinstance(chunk_id, str)
        and chunk_id.strip()
    }

    expected = {
        chunk_id.strip()
        for chunk_id in expected_chunk_ids
        if isinstance(chunk_id, str)
        and chunk_id.strip()
    }

    unique_cited = list(
        dict.fromkeys(cited)
    )

    invalid_citations = [
        chunk_id
        for chunk_id in unique_cited
        if chunk_id not in evidence
    ]

    valid_citation_count = (
        len(unique_cited)
        - len(invalid_citations)
    )

    if unique_cited:
        citation_validity = (
            valid_citation_count
            / len(unique_cited)
        )
    else:
        citation_validity = None

    if expected:
        cited_gold = (
            expected.intersection(
                unique_cited
            )
        )

        gold_citation_recall = (
            len(cited_gold)
            / len(expected)
        )
    else:
        cited_gold = set()
        gold_citation_recall = None

    return {
        "citation_count": len(
            unique_cited
        ),
        "valid_citation_count": (
            valid_citation_count
        ),
        "invalid_citations": (
            invalid_citations
        ),
        "citation_validity": (
            citation_validity
        ),
        "gold_chunk_count": len(
            expected
        ),
        "cited_gold_chunk_ids": sorted(
            cited_gold
        ),
        "gold_citation_recall": (
            gold_citation_recall
        ),
    }