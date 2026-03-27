"""
run_ingestion.py  —  src/ingestion/run_ingestion.py
─────────────────────────────────────────────────────
Entry point for the ingestion pipeline.

Run from project root:
    python -m src.ingestion.run_ingestion
"""

import sys
from pathlib import Path
import yaml

# MUST come before any src imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.pdf_loader import load_pdfs
from src.ingestion.chunker import chunk_pages_structured as chunk_pages
from src.ingestion.metadata_enricher import enrich_chunks
from src.ingestion.embedder import load_embedder, embed_chunks, unload_embedder
from src.ingestion.indexer import build_qdrant_client, create_collection, upsert_chunks


def run() -> None:
    # ---------------------------
    # Load config
    # ---------------------------
    cfg_path = PROJECT_ROOT / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    raw_docs_path = str(PROJECT_ROOT / cfg["data"]["raw_docs_path"])
    db_path = str(PROJECT_ROOT / "qdrant_db")

    print("\n" + "=" * 60)
    print("  INGESTION PIPELINE")
    print("=" * 60 + "\n")

    # ---------------------------
    # 1. Load PDFs
    # ---------------------------
    pages = load_pdfs(raw_docs_path)

    # ---------------------------
    # 2. Chunk
    # ---------------------------
    chunks = chunk_pages(
        pages,
        chunk_size=cfg["chunking"]["chunk_size"],
        overlap=cfg["chunking"]["overlap"],
    )

    # ---------------------------
    # 3. Enrich metadata
    # ---------------------------
    chunks = enrich_chunks(chunks)

    # ---------------------------
    # 4. Embed
    # ---------------------------
    embed_model = load_embedder(
        cfg["embedding"]["model_name"],
        cfg["embedding"]["device"],
    )

    chunks = embed_chunks(chunks, embed_model)
    unload_embedder(embed_model)

    # ---------------------------
    # 5. Index
    # ---------------------------
    client = build_qdrant_client(db_path)

    create_collection(
        client,
        cfg["qdrant"]["collection_name"],
        cfg["qdrant"]["vector_size"],
        recreate=True,
    )

    upsert_chunks(client, cfg["qdrant"]["collection_name"], chunks)

    # ---------------------------
    # Done
    # ---------------------------
    print("=" * 60)
    print(f"INGESTION COMPLETE — {len(chunks)} chunks indexed")
    print(f"Database: {db_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run()