"""Build reproducible local retrieval artifacts without any Gemini/Qdrant calls.

Run from this project's root, using Python 3.11:
    python -m pip install -r requirements-artifacts.txt
    python -m scripts.build_retrieval_artifacts

The manifest is local-only; compare it with earlier trusted builds before
using the BM25 index alongside an already populated Qdrant collection.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import platform

import bm25s
import pymupdf

from src.indexing.bm25_index import BM25Index
from src.indexing.chunk_loader import load_indexing_chunks
from src.ingestion.pipeline import process_and_export


EXPECTED_TOTAL = 1985
EXPECTED_BODY = 1766
EXPECTED_TABLE = 219
ARTIFACT_DIR = Path("data/processed")
BM25_DIR = Path("data/bm25")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    result = process_and_export()
    chunks = load_indexing_chunks()
    counts = Counter(chunk.content_type for chunk in chunks)

    actual = (len(chunks), counts["body"], counts["table"])
    expected = (EXPECTED_TOTAL, EXPECTED_BODY, EXPECTED_TABLE)
    if actual != expected:
        raise ValueError(f"Corpus count mismatch: expected {expected}, got {actual}")

    bm25 = BM25Index()
    count = bm25.build()
    if count != len(chunks):
        raise ValueError(f"BM25 indexed {count}, expected {len(chunks)}")
    bm25.save()

    reloaded = BM25Index()
    reloaded.load()
    if len(reloaded.search("SEC-IR-004 SEV-1 incident reporting", limit=5)) != 5:
        raise ValueError("BM25 reload/search validation failed")

    # Verify generated files before writing a manifest.
    files = [result.documents_path, result.tables_path, result.chunks_path]
    source_pdfs = sorted(Path("data/raw").glob("*.pdf"))
    if len(source_pdfs) != 12:
        raise ValueError(f"Expected 12 input PDFs, got {len(source_pdfs)}")

    manifest = {
        "python": platform.python_version(),
        "pymupdf": pymupdf.VersionBind,
        "bm25s": getattr(bm25s, "__version__", "unknown"),
        "chunk_count": len(chunks),
        "body_chunks": counts["body"],
        "table_chunks": counts["table"],
        "source_pdf_sha256": {
            path.name: _sha256_file(path) for path in source_pdfs
        },
        "generated_sha256": {
            str(path): _sha256_file(path) for path in files
        },
        "warning": (
            "Count equality is not Qdrant index equivalence. "
            "Validate chunk IDs and text against stored Qdrant payloads "
            "before using the new artifacts for production retrieval."
        ),
    }
    manifest_path = ARTIFACT_DIR / "retrieval_artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Built {count} BM25 entries; manifest: {manifest_path}")
    print(f"Chunk file SHA256: {manifest['generated_sha256'][str(result.chunks_path)]}")
    print("Production Qdrant alignment remains UNVERIFIED.")


if __name__ == "__main__":
    main()
