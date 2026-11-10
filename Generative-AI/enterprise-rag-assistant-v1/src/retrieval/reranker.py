from typing import Any

from sentence_transformers import CrossEncoder


RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)


class CrossEncoderReranker:
    """
    Reranks retrieval candidates by directly
    scoring each query + chunk-text pair.
    """

    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
    ) -> None:
        self.model = CrossEncoder(
            model_name
        )

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        """
        Score every query/chunk pair and return
        candidates ordered by reranker relevance.
        """

        if not query.strip():
            raise ValueError(
                "Reranker query cannot be empty"
            )

        if not candidates:
            return []

        if limit <= 0:
            raise ValueError(
                "Reranker limit must be greater than zero"
            )

        pairs = []

        for candidate in candidates:
            chunk = candidate["chunk"]

            text = chunk.get("text")

            if not text:
                raise ValueError(
                    "Candidate chunk contains no text"
                )

            pairs.append(
                [
                    query,
                    text,
                ]
            )

        scores = self.model.predict(
            pairs
        )

        reranked_results = []

        for candidate, score in zip(
            candidates,
            scores,
        ):
            result = dict(candidate)

            result["reranker_score"] = float(
                score
            )

            reranked_results.append(
                result
            )

        reranked_results.sort(
            key=lambda result: (
                result["reranker_score"]
            ),
            reverse=True,
        )

        reranked_results = reranked_results[
            :limit
        ]

        for rank, result in enumerate(
            reranked_results,
            start=1,
        ):
            result["reranker_rank"] = rank

        return reranked_results