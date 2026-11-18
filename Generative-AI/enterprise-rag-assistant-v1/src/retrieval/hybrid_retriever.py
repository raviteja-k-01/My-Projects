from typing import Any

from google.genai import errors as genai_errors

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
        """Strict hybrid retrieval for evaluation.

        NEVER silently downgrade evaluation to BM25, or hybrid accuracy metrics
        would be misleading.
        """
        ranked, _mode = self.retrieve_with_mode(
            query=query,
            dense_limit=dense_limit,
            sparse_limit=sparse_limit,
            fused_limit=fused_limit,
            allow_sparse_fallback=False,
        )
        return ranked

    def retrieve_with_mode(
        self,
        query: str,
        dense_limit: int = DEFAULT_DENSE_LIMIT,
        sparse_limit: int = DEFAULT_SPARSE_LIMIT,
        fused_limit: int = DEFAULT_FUSED_LIMIT,
        *,
        allow_sparse_fallback: bool = False,
    ) -> tuple[list[dict[str, Any]], str]:
        """Return (ranked_chunks, mode), with explicit application-only fallback.

        When enabled, only Gemini 429 and 5xx service errors downgrade to BM25.
        Invalid credentials, bad vectors, code bugs and Qdrant errors still fail.
        """
        if not query.strip():
            raise ValueError("Retrieval query cannot be empty")

        mode = "hybrid"
        try:
            dense_results = self._dense_search(query=query, limit=dense_limit)
        except genai_errors.ClientError as exc:
            if not allow_sparse_fallback or getattr(exc, "code", None) != 429:
                raise
            dense_results = []
            mode = "bm25_fallback"
        except genai_errors.ServerError as exc:
            if not allow_sparse_fallback or getattr(exc, "code", None) not in (500, 502, 503, 504):
                raise
            dense_results = []
            mode = "bm25_fallback"

        sparse_results = self._sparse_search(query=query, limit=sparse_limit)
        ranked = reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            limit=fused_limit,
        )
        return ranked, mode

    def close(self) -> None:
        """
        Close the Qdrant connection.
        """

        self.qdrant.close()