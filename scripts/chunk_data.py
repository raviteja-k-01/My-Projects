
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from app.config import CHUNK_SIZE, CHUNK_OVERLAP

CLEAN_PATH = "data/cleaned"
OUTPUT_FILE = "data/processed/chunks.json"

def chunk_text(text, chunk_size=512, overlap=75):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))

        start += (chunk_size - overlap)

    return chunks

def process():
    all_chunks = []
    chunk_id = 0

    for category in ["aws_docs", "internal_docs"]:
        folder = os.path.join(CLEAN_PATH, category)

        for file in os.listdir(folder):
            if not file.endswith(".txt"):
                continue

            file_path = os.path.join(folder, file)

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

            for chunk in chunks:
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "source": file.replace(".txt", ""),
                    "doc_type": category
                })
                chunk_id += 1

    os.makedirs("data/processed", exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

if __name__ == "__main__":
    process()