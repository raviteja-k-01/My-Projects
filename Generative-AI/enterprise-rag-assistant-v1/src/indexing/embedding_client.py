"""Gemini embeddings for the fixed 768-dimensional Qdrant collection.

One model and one embedding space are used regardless of which explicitly
selected credential authenticates the request. Credentials are never logged.
"""
from __future__ import annotations

from functools import lru_cache
import math
import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types


EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSION = 768

# Explicit credential selection. We do NOT silently rotate free-tier keys to
# bypass rate limits; Google API terms prohibit circumventing API limitations.
ALLOWED_EMBEDDING_KEY_ENVS = frozenset({
    "GEMINI_API_KEY",
    "GEMINI_API_KEY_1",
})


class GeminiEmbeddingClient:
    """Embedding client with bounded, process-local query cache.

    Documents are never cached (the ingestion script checkpoints in Qdrant).
    Queries are cached by exact stripped text within this process, avoiding
    repeated API requests for identical questions. No user data is persisted.
    """

    def __init__(self, *, key_env: str | None = None, client: Any | None = None) -> None:
        load_dotenv()
        selected = key_env or os.getenv("GEMINI_EMBEDDING_KEY_ENV", "GEMINI_API_KEY")
        if selected not in ALLOWED_EMBEDDING_KEY_ENVS:
            raise ValueError("Invalid GEMINI_EMBEDDING_KEY_ENV; select an allowed key name")

        self.key_env = selected
        if client is not None:
            # Injection for deterministic offline unit tests.
            self.client = client
        else:
            api_key = os.getenv(selected)
            if not api_key:
                raise ValueError(f"{selected} is missing")
            self.client = genai.Client(api_key=api_key)
        # Per-instance cache, not a global class cache retaining old clients.
        self._query_cache = lru_cache(maxsize=512)(self._embed_query_uncached)

    @staticmethod
    def _extract_vectors(response: Any, expected_count: int) -> list[list[float]]:
        if not response.embeddings:
            raise ValueError("Gemini returned no embeddings")
        if len(response.embeddings) != expected_count:
            raise ValueError(
                "Embedding count does not match input count: "
                f"expected {expected_count}, got {len(response.embeddings)}"
            )

        vectors: list[list[float]] = []
        for embedding in response.embeddings:
            if embedding.values is None:
                raise ValueError("Embedding contains no values")
            vector = [float(value) for value in embedding.values]
            if len(vector) != EMBEDDING_DIMENSION:
                raise ValueError(f"Unexpected embedding dimension: {len(vector)}")
            if not all(math.isfinite(value) for value in vector):
                raise ValueError("Embedding contains non-finite values")
            vectors.append(vector)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("Document text cannot be empty")

        contents = [
            types.Content(
                parts=[types.Part.from_text(text=f"title: none | text: {text}")]
            )
            for text in texts
        ]
        response = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=contents,
            config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSION),
        )
        return self._extract_vectors(response, len(texts))

    def _embed_query_uncached(self, clean_text: str) -> tuple[float, ...]:
        """Only a completed successful response is cached."""
        prepared_query = f"task: question answering | query: {clean_text}"
        response = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=prepared_query,
            config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSION),
        )
        return tuple(self._extract_vectors(response, 1)[0])

    def embed_query(self, text: str) -> list[float]:
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Query text cannot be empty")
        # Defensive copy: caller cannot mutate a cached vector.
        return list(self._query_cache(clean_text))
