from typing import Any


DEFAULT_K_VALUES = (1, 3, 5, 10, 20)


def evaluate_ranked_chunks(
    retrieved_chunk_ids: list[str],
    expected_chunk_ids: list[str],
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> dict[str, Any]:
    """
    Evaluate one ranked retrieval result against
    the expected gold chunk IDs.

    Metrics:
    - Hit@K
    - Recall@K
    - MRR (Mean Reciprocal Rank contribution)

    Unanswerable questions with no gold chunks are
    marked as not evaluable for retrieval.
    """

    retrieved = [
        chunk_id.strip()
        for chunk_id in retrieved_chunk_ids
        if isinstance(chunk_id, str)
        and chunk_id.strip()
    ]

    expected = {
        chunk_id.strip()
        for chunk_id in expected_chunk_ids
        if isinstance(chunk_id, str)
        and chunk_id.strip()
    }

    if not expected:
        return {
            "evaluable": False,
            "gold_chunk_count": 0,
            "first_relevant_rank": None,
            "mrr": None,
            "hit_at_k": {},
            "recall_at_k": {},
        }

    first_relevant_rank = None

    for rank, chunk_id in enumerate(
        retrieved,
        start=1,
    ):
        if chunk_id in expected:
            first_relevant_rank = rank
            break

    if first_relevant_rank is None:
        reciprocal_rank = 0.0
    else:
        reciprocal_rank = (
            1.0 / first_relevant_rank
        )

    hit_at_k: dict[str, float] = {}
    recall_at_k: dict[str, float] = {}

    for k in k_values:
        top_k = set(
            retrieved[:k]
        )

        relevant_found = (
            expected.intersection(top_k)
        )

        hit_at_k[f"hit@{k}"] = (
            1.0
            if relevant_found
            else 0.0
        )

        recall_at_k[f"recall@{k}"] = (
            len(relevant_found)
            / len(expected)
        )

    return {
        "evaluable": True,
        "gold_chunk_count": len(expected),
        "first_relevant_rank": (
            first_relevant_rank
        ),
        "mrr": reciprocal_rank,
        "hit_at_k": hit_at_k,
        "recall_at_k": recall_at_k,
    }