from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class Reranker:
    def __init__(self):
        self.model = CrossEncoder(MODEL_NAME)

    def rerank(self, query, results, top_k=5):
        # results: [{"text": ..., "source": ..., "score": ...}, ...]
        pairs = [[query, r["text"]] for r in results]

        scores = self.model.predict(pairs)

        for i, r in enumerate(results):
            r["rerank_score"] = float(scores[i])

        ranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

        return ranked[:top_k]