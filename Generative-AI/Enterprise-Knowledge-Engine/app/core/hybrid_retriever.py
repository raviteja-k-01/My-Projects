import json
from app.core.bm25 import BM25Retriever
from app.core.dense import DenseRetriever
from app.core.fusion import fuse

CHUNKS_FILE = "data/processed/chunks.json"

class HybridRetriever:
    def __init__(self):
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever()

        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    def search(self, query, top_k=5):
        bm25_results = self.bm25.search(query, top_k=20)
        dense_results = self.dense.search(query, top_k=20)

        fused = fuse(bm25_results, dense_results)

        results = []
        for idx, score in fused[:top_k]:
            results.append({
                "text": self.chunks[idx]["text"],
                "source": self.chunks[idx]["source"],
                "score": score
            })

        return results