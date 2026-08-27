import math
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSION = 768


class GeminiEmbeddingClient:
    """
    Creates Gemini embeddings for:

    1. Enterprise document chunks during indexing.
    2. User questions during dense retrieval.
    """

    def __init__(self) -> None:
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing"
            )

        self.client = genai.Client(
            api_key=api_key
        )

    def _extract_vectors(
        self,
        response,
        expected_count: int,
    ) -> list[list[float]]:
        """
        Validate Gemini's response and convert
        embedding values into normal Python lists.
        """

        if not response.embeddings:
            raise ValueError(
                "Gemini returned no embeddings"
            )

        if len(response.embeddings) != expected_count:
            raise ValueError(
                "Embedding count does not match "
                f"input count. Expected "
                f"{expected_count}, got "
                f"{len(response.embeddings)}"
            )

        vectors: list[list[float]] = []

        for embedding in response.embeddings:

            if embedding.values is None:
                raise ValueError(
                    "Embedding contains no values"
                )

            vector = [
                float(value)
                for value in embedding.values
            ]

            if len(vector) != EMBEDDING_DIMENSION:
                raise ValueError(
                    "Unexpected embedding dimension: "
                    f"{len(vector)}"
                )

            if not all(
                math.isfinite(value)
                for value in vector
            ):
                raise ValueError(
                    "Embedding contains "
                    "non-finite values"
                )

            vectors.append(vector)

        return vectors

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Convert document chunks into separate
        embeddings for storage in Qdrant.
        """

        if not texts:
            raise ValueError(
                "At least one document text "
                "is required"
            )

        if any(
            not text.strip()
            for text in texts
        ):
            raise ValueError(
                "Document text cannot be empty"
            )

        contents = [
            types.Content(
                parts=[
                    types.Part.from_text(
                        text=(
                            "title: none | "
                            f"text: {text}"
                        )
                    )
                ]
            )
            for text in texts
        ]

        response = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=contents,
            config=types.EmbedContentConfig(
                output_dimensionality=(
                    EMBEDDING_DIMENSION
                )
            ),
        )

        return self._extract_vectors(
            response=response,
            expected_count=len(texts),
        )

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """
        Convert one user's question into an
        embedding for dense retrieval.
        """

        if not text.strip():
            raise ValueError(
                "Query text cannot be empty"
            )

        prepared_query = (
            "task: question answering | "
            f"query: {text}"
        )

        response = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=prepared_query,
            config=types.EmbedContentConfig(
                output_dimensionality=(
                    EMBEDDING_DIMENSION
                )
            ),
        )

        vectors = self._extract_vectors(
            response=response,
            expected_count=1,
        )

        return vectors[0]