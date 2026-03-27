import time
from typing import List, Dict

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


def dense_search(
    query: str,
    client: QdrantClient,
    embed_model: SentenceTransformer,
    collection_name: str,
    top_k: int = 20
) -> List[Dict]:
    """
    Perform dense vector search using Qdrant.
    """

    start = time.time()

    # 1. Convert query to embedding
    query_vector = embed_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    # 2. Search Qdrant
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True
    )

    # 3. Format results
    chunks = []

    for r in results:
        payload = r.payload

        payload["score"] = float(r.score)
        payload["retrieval_type"] = "dense"

        chunks.append(payload)

    latency_ms = int((time.time() - start) * 1000)

    print(f"[DENSE] {len(chunks)} results (top score: {chunks[0]['score']:.3f})")

    return chunks