# src/ingestion/__init__.py
from .pdf_loader        import load_pdfs
from .chunker import chunk_pages_structured as chunk_pages
from .metadata_enricher import enrich_chunks
from .embedder          import load_embedder, embed_chunks, unload_embedder
from .indexer           import build_qdrant_client, create_collection, upsert_chunks
