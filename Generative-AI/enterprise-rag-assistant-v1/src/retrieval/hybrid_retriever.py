from typing import Any

from src.indexing.bm25_index import BM25Index
from src.indexing.embedding_client import GeminiEmbeddingClient
from src.indexing.qdrant_store import QdrantStore
from src.retrieval.rrf import reciprocal_rank_fusion


DEFAULT_DENSE_LIMIT = 20
DEFAULT_SPARSE_LIMIT = 20
DEFAULT_FUSED_LIMIT = 20


class HybridRetriever:
    """
    Runs dense Qdrant retrieval and sparse BM25
    retrieval, then combines both rankings with RRF.
    """

    def __init__(self) -> None:
        self.embedder = GeminiEmbeddingClient()
        self.qdrant = QdrantStore()

        self.bm25 = BM25Index()
        self.bm25.load()

    def _dense_search(
        self,
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """
        Embed the user query and retrieve the nearest
        chunks from Qdrant Cloud.
        """

        query_vector = self.embedder.embed_query(
            query
        )

        results = self.qdrant.query(
            query_vector=query_vector,
            limit=limit,
        )

        normalized_results: list[
            dict[str, Any]
        ] = []

        for rank, result in enumerate(
            results,
            start=1,
        ):
            if result.payload is None:
                raise ValueError(
                    f"Qdrant point {result.id} "
                    "contains no payload"
                )

            chunk = dict(result.payload)

            chunk_id = chunk.get(
                "chunk_id"
            )

            if not chunk_id:
                raise ValueError(
                    f"Qdrant point {result.id} "
                    "contains no chunk_id"
                )

            normalized_results.append(
                {
                    "chunk_id": chunk_id,
                    "rank": rank,
                    "score": float(
                        result.score
                    ),
                    "chunk": chunk,
                }
            )

        return normalized_results

    def _sparse_search(
        self,
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """
        Retrieve keyword-relevant chunks using BM25.
        """

        results = self.bm25.search(
            query=query,
            limit=limit,
        )

        normalized_results: list[
            dict[str, Any]
        ] = []

        for result in results:
            chunk = result["chunk"]

            chunk_id = chunk.get(
                "chunk_id"
            )

            if not chunk_id:
                raise ValueError(
                    "BM25 result contains "
                    "no chunk_id"
                )

            normalized_results.append(
                {
                    "chunk_id": chunk_id,
                    "rank": result["rank"],
                    "score": result["score"],
                    "chunk": chunk,
                }
            )

        return normalized_results

    def retrieve(
        self,
        query: str,
        dense_limit: int = DEFAULT_DENSE_LIMIT,
        sparse_limit: int = DEFAULT_SPARSE_LIMIT,
        fused_limit: int = DEFAULT_FUSED_LIMIT,
    ) -> list[dict[str, Any]]:
        """
        Run dense + sparse retrieval and combine
        the rankings with Reciprocal Rank Fusion.
        """

        if not query.strip():
            raise ValueError(
                "Retrieval query cannot be empty"
            )

        dense_results = self._dense_search(
            query=query,
            limit=dense_limit,
        )

        sparse_results = self._sparse_search(
            query=query,
            limit=sparse_limit,
        )

        fused_results = reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            limit=fused_limit,
        )

        return fused_results

    def close(self) -> None:
        """
        Close the Qdrant connection.
        """

        self.qdrant.close()