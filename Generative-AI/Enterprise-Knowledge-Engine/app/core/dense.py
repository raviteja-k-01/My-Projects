import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "BAAI/bge-base-en-v1.5"
INDEX_FILE = "data/index/faiss.index"


class DenseRetriever:
    def __init__(self):
        # Load tokenizer + model (lighter than sentence-transformers wrapper)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)

        # Load FAISS index
        self.index = faiss.read_index(INDEX_FILE)

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings.cpu().numpy()

    def search(self, query, top_k=10):
        query_vec = self.encode(query)

        distances, indices = self.index.search(
            np.array(query_vec).astype("float32"),
            top_k
        )

        return list(zip(indices[0], distances[0]))