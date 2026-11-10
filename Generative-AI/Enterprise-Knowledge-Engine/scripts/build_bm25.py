import json

INPUT_FILE = "data/processed/chunks.json"
OUTPUT_FILE = "data/processed/bm25_corpus.json"

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    corpus = [chunk["text"] for chunk in chunks]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2)

if __name__ == "__main__":
    main()