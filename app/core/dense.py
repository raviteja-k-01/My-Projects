import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-base-en-v1.5"
INDEX_FILE = "data/index/faiss.index"

class DenseRetriever:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(INDEX_FILE)

    def search(self, query, top_k=10):
        query_vec = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_vec), top_k)

        return list(zip(indices[0], distances[0]))