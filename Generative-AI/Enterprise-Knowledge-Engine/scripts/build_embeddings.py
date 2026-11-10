import json
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-base-en-v1.5"

INPUT_FILE = "data/processed/chunks.json"
OUTPUT_FILE = "data/processed/embeddings.npy"

def main():
    model = SentenceTransformer(MODEL_NAME)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(texts, show_progress_bar=True)

    np.save(OUTPUT_FILE, embeddings)

if __name__ == "__main__":
    main()