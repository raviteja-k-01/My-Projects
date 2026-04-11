import time
from app.core.hybrid_retriever import HybridRetriever
from app.core.reranker import Reranker
from app.core.llm import generate_answer

class QueryService:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.reranker = Reranker()

    def query(self, query):
        start_time = time.time()

        # Step 1: retrieve
        retrieved = self.retriever.search(query, top_k=20)

        # Step 2: rerank
        reranked = self.reranker.rerank(query, retrieved, top_k=5)

        # Step 3: prepare context
        contexts = [r["text"] for r in reranked]

        # Step 4: LLM answer
        answer = generate_answer(query, contexts)

        latency = time.time() - start_time

        return {
            "query": query,
            "answer": answer,
            "sources": reranked,
            "latency": latency
        }