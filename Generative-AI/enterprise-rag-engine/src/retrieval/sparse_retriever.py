import re
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient


def _tokenize(text: str) -> List[str]:
    """
    Simple tokenizer:
    - lowercase
    - split words and numbers
    """
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    """
    Build BM25 index from Qdrant payloads (once at startup)
    """

    def __init__(self, client: QdrantClient, collection_name: str):
        print(f"[BM25] Building index from '{collection_name}'...")

        self.chunks: List[Dict[str, Any]] = []

        offset = None

        while True:
            results, offset = client.scroll(
                collection_name=collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )

            for point in results:
                self.chunks.append(point.payload)

            if offset is None:
                break

        if not self.chunks:
            raise ValueError("BM25: No data found in Qdrant")

        tokenized = [_tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

        print(f"[BM25] Index ready ({len(self.chunks)} chunks)\n")

    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        tokens = _tokenize(query)

        scores = self.bm25.get_scores(tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        max_score = max(scores) if max(scores) > 0 else 1.0

        results = []

        for idx in ranked_indices:
            if scores[idx] <= 0:
                break

            chunk = self.chunks[idx].copy()
            chunk["score"] = float(scores[idx] / max_score)
            chunk["retrieval_type"] = "sparse"

            results.append(chunk)

        print(f"[BM25] {len(results)} results (top score: {results[0]['score']:.3f})")

        return results