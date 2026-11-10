import json
from rank_bm25 import BM25Okapi

CORPUS_FILE = "data/processed/bm25_corpus.json"

class BM25Retriever:
    def __init__(self):
        with open(CORPUS_FILE, "r", encoding="utf-8") as f:
            corpus = json.load(f)

        self.tokenized_corpus = [doc.split() for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query, top_k=10):
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            list(enumerate(scores)),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:top_k]