from pathlib import Path

from src.core.models import ChunkRecord
from src.ingestion.exporter import load_chunks


DEFAULT_CHUNKS_PATH = Path(
    "data/processed/chunks.jsonl"
)


def load_indexing_chunks(
    path: Path = DEFAULT_CHUNKS_PATH,
) -> list[ChunkRecord]:
    """
    Load the persisted retrieval-ready chunks that will
    feed both dense indexing and BM25 indexing.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"Chunks file not found: {path}"
        )

    chunks = load_chunks(path)

    if not chunks:
        raise ValueError(
            "No chunks were loaded for indexing"
        )

    seen_chunk_ids: set[str] = set()

    for chunk in chunks:
        if not chunk.chunk_id.strip():
            raise ValueError(
                "Chunk ID cannot be empty"
            )

        if chunk.chunk_id in seen_chunk_ids:
            raise ValueError(
                f"Duplicate chunk ID: "
                f"{chunk.chunk_id}"
            )

        seen_chunk_ids.add(
            chunk.chunk_id
        )

        if not chunk.text.strip():
            raise ValueError(
                f"Chunk text cannot be empty: "
                f"{chunk.chunk_id}"
            )

        if chunk.token_count <= 0:
            raise ValueError(
                f"Invalid token count: "
                f"{chunk.chunk_id}"
            )

        if chunk.content_type not in {
            "body",
            "table",
        }:
            raise ValueError(
                f"Unsupported content type: "
                f"{chunk.chunk_id}"
            )

    return chunks