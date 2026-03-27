# src/evaluation/test_set_builder.py

import json
import time
import yaml
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

from src.retrieval.retriever import Retriever
from src.generation.generator import generate_answer


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config():
    with open(PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def load_test_set(path):
    with open(path) as f:
        return json.load(f)


def run():
    cfg = load_config()

    test_set = load_test_set(PROJECT_ROOT / cfg["evaluation"]["test_set_path"])

    client = QdrantClient(path=str(PROJECT_ROOT / "qdrant_db"))

    embed_model = SentenceTransformer(cfg["embedding"]["model_name"])
    rerank_model = CrossEncoder(cfg["reranker"]["model_name"])

    retriever = Retriever(client, embed_model, rerank_model, cfg)

    outputs = []

    for i, item in enumerate(test_set, 1):
        print(f"[{i}/{len(test_set)}] {item['question']}")

        start = time.time()

        retrieval = retriever.retrieve(item["question"])

        gen = generate_answer(
            item["question"],
            retrieval["chunks"],
            cfg
        )

        outputs.append({
                "question": item["question"].strip(),
                "ground_truths": [item["ground_truth"]],
                "answer": gen["answer"],
                "contexts": [c.get("text", "") for c in retrieval["chunks"] if c.get("text")],
            })

        print(f"   Done in {int(time.time()-start)}s")

    out_path = PROJECT_ROOT / "data/eval/rag_outputs.json"

    with open(out_path, "w") as f:
        json.dump(outputs, f, indent=2)

    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    run()    