import math
import os
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from src.indexing.embedding_client import EMBEDDING_DIMENSION


COLLECTION_NAME = "enterprise_rag_chunks"


class QdrantStore:
    """
    Handles dense-vector storage and retrieval
    using Qdrant Cloud.
    """

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        load_dotenv()

        self.collection_name = collection_name

        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url:
            raise ValueError(
                "QDRANT_URL is missing"
            )

        if not qdrant_api_key:
            raise ValueError(
                "QDRANT_API_KEY is missing"
            )

        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )

    def ensure_collection(self) -> None:
        if self.client.collection_exists(
            self.collection_name
        ):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=models.Distance.COSINE,
            ),
        )

    def _validate_vector(
        self,
        vector: list[float],
    ) -> None:

        if len(vector) != EMBEDDING_DIMENSION:
            raise ValueError(
                "Unexpected vector dimension: "
                f"{len(vector)}. "
                f"Expected {EMBEDDING_DIMENSION}."
            )

        if not all(
            math.isfinite(value)
            for value in vector
        ):
            raise ValueError(
                "Vector contains non-finite values"
            )

    def upsert_points(
        self,
        point_ids: list[int],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:

        if not point_ids:
            raise ValueError(
                "At least one point is required"
            )

        if not (
            len(point_ids)
            == len(vectors)
            == len(payloads)
        ):
            raise ValueError(
                "Point IDs, vectors, and payloads "
                "must have the same length"
            )

        points: list[models.PointStruct] = []

        for point_id, vector, payload in zip(
            point_ids,
            vectors,
            payloads,
        ):
            self._validate_vector(vector)

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def query(
        self,
        query_vector: list[float],
        limit: int = 20,
    ) -> list[models.ScoredPoint]:

        self._validate_vector(query_vector)

        if limit <= 0:
            raise ValueError(
                "Query limit must be greater than zero"
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        return response.points

    def count(self) -> int:
        result = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        )

        return result.count

    def close(self) -> None:
        self.client.close()