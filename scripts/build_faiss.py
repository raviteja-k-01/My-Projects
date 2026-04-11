import json
import faiss
import numpy as np

EMBEDDINGS_FILE = "data/processed/embeddings.npy"
CHUNKS_FILE = "data/processed/chunks.json"

INDEX_FILE = "data/index/faiss.index"
MAPPING_FILE = "data/index/id_mapping.json"

def main():
    embeddings = np.load(EMBEDDINGS_FILE)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)

    # mapping FAISS id → chunk_id
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    id_mapping = {i: chunks[i]["chunk_id"] for i in range(len(chunks))}

    with open(MAPPING_FILE, "w") as f:
        json.dump(id_mapping, f)

if __name__ == "__main__":
    main()