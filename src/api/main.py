import time
from pathlib import Path
from typing import List, Dict, Any

import yaml
from fastapi import FastAPI
from pydantic import BaseModel

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

from src.retrieval.retriever import Retriever
from src.generation.generator import generate_answer


# -------------------------
# Request schema
# -------------------------
class QueryRequest(BaseModel):
    question: str


# -------------------------
# App setup
# -------------------------
app = FastAPI()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

retriever = None
cfg = None


# -------------------------
# Load config
# -------------------------
def load_config():
    with open(PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


# -------------------------
# Startup (load models once)
# -------------------------
@app.on_event("startup")
def startup():
    global retriever, cfg

    print("\n[API] Starting pipeline...")

    cfg = load_config()

    # Qdrant client
    client = QdrantClient(path=str(PROJECT_ROOT / "qdrant_db"))

    # Models
    embed_model = SentenceTransformer(cfg["embedding"]["model_name"])
    rerank_model = CrossEncoder(cfg["reranker"]["model_name"])

    # Retriever
    retriever = Retriever(client, embed_model, rerank_model, cfg)

    print("[API] Ready\n")

    print(f"[API] Using model: {cfg['llm']['model']}")
# -------------------------
# Health check
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -------------------------
# Query endpoint
# -------------------------
@app.post("/query")

def query(req: QueryRequest):
    global retriever, cfg

    question = (req.question or "").strip()
    if retriever is None:
        return {
            "answer": "System not ready. Please try again later.",
            "citations": [],
            "confidence": "low",
            "latency_ms": 0,
            "chunk_count": 0,
        }
    # Basic guard (small talk)
    if len(question) < 10:
        return {
            "answer": "Ask a policy-related question.",
            "citations": [],
            "confidence": "low",
            "latency_ms": 0,
            "chunk_count": 0,
        }

    # Retrieve
    try:
        retrieval = retriever.retrieve(question)

        result = generate_answer(
            question,
            retrieval.get("chunks", []),
            cfg
        )

        return result

    except Exception as e:
        print(f"[API ERROR] {e}")

        return {
            "answer": "Internal error. Please try again.",
            "citations": [],
            "confidence": "low",
            "latency_ms": 0,
            "chunk_count": 0,
        }