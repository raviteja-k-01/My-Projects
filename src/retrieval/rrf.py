from typing import Any


RRF_K = 60


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    limit: int | None = None,
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    """
    Combine dense and sparse ranked results using
    Reciprocal Rank Fusion (RRF).

    Each input result must contain:
        chunk_id
        rank
        score
        chunk
    """

    if k <= 0:
        raise ValueError(
            "RRF k must be greater than zero"
        )

    fused: dict[str, dict[str, Any]] = {}

    def add_results(
        results: list[dict[str, Any]],
        source: str,
    ) -> None:

        for result in results:
            chunk_id = result["chunk_id"]
            rank = result["rank"]

            if rank <= 0:
                raise ValueError(
                    "Result rank must be greater than zero"
                )

            rrf_score = 1.0 / (k + rank)

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "chunk_id": chunk_id,
                    "chunk": result["chunk"],
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "dense_score": None,
                    "sparse_rank": None,
                    "sparse_score": None,
                }

            fused[chunk_id]["rrf_score"] += rrf_score

            if source == "dense":
                fused[chunk_id]["dense_rank"] = rank
                fused[chunk_id]["dense_score"] = (
                    result["score"]
                )

            elif source == "sparse":
                fused[chunk_id]["sparse_rank"] = rank
                fused[chunk_id]["sparse_score"] = (
                    result["score"]
                )

    add_results(
        results=dense_results,
        source="dense",
    )

    add_results(
        results=sparse_results,
        source="sparse",
    )

    ranked_results = sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    for final_rank, result in enumerate(
        ranked_results,
        start=1,
    ):
        result["rank"] = final_rank

    if limit is not None:
        if limit <= 0:
            raise ValueError(
                "RRF limit must be greater than zero"
            )

        ranked_results = ranked_results[:limit]

    return ranked_results