import time
from typing import Dict, Any

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

from src.retrieval.dense_retriever import dense_search
from src.retrieval.sparse_retriever import BM25Index
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.deduplicator import deduplicate
from src.retrieval.reranker import rerank


class Retriever:
    def __init__(
        self,
        qdrant_client: QdrantClient,
        embed_model: SentenceTransformer,
        rerank_model: CrossEncoder,
        cfg: dict
    ):
        self.client = qdrant_client
        self.embed_model = embed_model
        self.rerank_model = rerank_model

        self.collection_name = cfg["qdrant"]["collection_name"]

        self.dense_top_k = cfg["retrieval"]["dense_top_k"]
        self.sparse_top_k = cfg["retrieval"]["sparse_top_k"]
        self.fusion_top_k = cfg["retrieval"]["fusion_top_k"]

        self.bm25 = BM25Index(self.client, self.collection_name)

    def retrieve(self, query: str) -> Dict[str, Any]:
        start = time.time()

        # -------------------------
        # 1. Dense retrieval
        # -------------------------
        dense_chunks = dense_search(
            query,
            self.client,
            self.embed_model,
            self.collection_name,
            top_k=self.dense_top_k
        )

        # -------------------------
        # 2. Sparse retrieval
        # -------------------------
        sparse_chunks = self.bm25.search(
            query,
            top_k=self.sparse_top_k
        )

        # -------------------------
        # 3. Fusion
        # -------------------------
        fused_chunks = reciprocal_rank_fusion(
            dense_chunks,
            sparse_chunks,
            top_k=self.fusion_top_k
        )

        # -------------------------
        # 4. Deduplicate
        # -------------------------
        fused_chunks = deduplicate(fused_chunks, threshold=0.85)

        # -------------------------
        # 5. Heuristic scoring
        # -------------------------
        def boost(c):
            score = 0
            text = c["text"].lower()

            if c.get("type") == "table_row":
                score += 2

            if any(x in text for x in ["must", "shall", "limit", "maximum", "minimum"]):
                score += 1

            if any(x in text for x in ["₹", "rs", "days", "hours", "%"]):
                score += 1

            return score

        fused_chunks = sorted(fused_chunks, key=boost, reverse=True)

        # -------------------------
        # 6. Query-aware boosting
        # -------------------------
        if any(x in query.lower() for x in ["tier", "limit", "cost", "hours", "band"]):
            fused_chunks = sorted(
                fused_chunks,
                key=lambda c: 1 if c.get("type") == "table_row" else 0,
                reverse=True
            )

        # -------------------------
        # 7. Candidate selection
        # -------------------------
        candidates = fused_chunks[:20]

        candidates = sorted(
            candidates,
            key=lambda c: abs(c.get("length", 200) - 300)
        )[:15]

        # -------------------------
        # 8. Rerank
        # -------------------------
        if candidates:
            top_chunks = rerank(
                query,
                candidates,
                self.rerank_model,
                top_k=5
            )
        else:
            top_chunks = []

        latency_ms = int((time.time() - start) * 1000)

        return {
            "chunks": top_chunks,
            "latency_ms": latency_ms,
            "rewritten_query": query
        }