import time
from dataclasses import asdict

from google.genai import errors

from src.indexing.chunk_loader import load_indexing_chunks
from src.indexing.embedding_client import GeminiEmbeddingClient
from src.indexing.qdrant_store import QdrantStore


BATCH_SIZE = 16

# 16 embeddings every 10.5 seconds
# keeps us below roughly 100 embeddings/minute.
BATCH_DELAY_SECONDS = 10.5

# Fallback wait if Gemini still returns 429.
RATE_LIMIT_RETRY_SECONDS = 65


def embed_with_retry(
    embedder: GeminiEmbeddingClient,
    texts: list[str],
) -> list[list[float]]:

    while True:
        try:
            return embedder.embed_documents(texts)

        except errors.ClientError as exc:

            if exc.code != 429:
                raise

            error_text = str(exc)

            # Daily free-tier quota exhausted.
            if (
                "PerDay" in error_text
                or "requests per day" in error_text.lower()
            ):
                raise RuntimeError(
                    "\nGemini daily embedding quota exhausted.\n"
                    "The vectors already stored in Qdrant are safe.\n"
                    "Run this script again after the daily quota resets."
                ) from exc

            # Temporary per-minute quota.
            print(
                "\nGemini per-minute rate limit reached."
            )

            print(
                f"Waiting "
                f"{RATE_LIMIT_RETRY_SECONDS} seconds..."
            )

            time.sleep(
                RATE_LIMIT_RETRY_SECONDS
            )

def main() -> None:
    chunks = load_indexing_chunks()

    total_chunks = len(chunks)

    if total_chunks == 0:
        raise ValueError(
            "No chunks available for indexing"
        )

    embedder = GeminiEmbeddingClient()
    store = QdrantStore()

    try:
        store.ensure_collection()

        # Existing points allow interrupted builds
        # to continue instead of starting over.
        existing_count = store.count()

        if existing_count > total_chunks:
            raise ValueError(
                "Qdrant contains more points than "
                "the current chunk dataset."
            )

        print("\nDENSE INDEX BUILD")
        print(
            f"Total chunks: {total_chunks}"
        )
        print(
            f"Already indexed: {existing_count}"
        )
        print(
            f"Remaining: "
            f"{total_chunks - existing_count}"
        )
        print(
            f"Batch size: {BATCH_SIZE}"
        )

        if existing_count == total_chunks:
            print(
                "\nAll chunks are already indexed."
            )
            print(
                "DENSE INDEX BUILD: PASS"
            )
            return

        # Resume exactly after the points that
        # have already been successfully stored.
        for start in range(
            existing_count,
            total_chunks,
            BATCH_SIZE,
        ):
            end = min(
                start + BATCH_SIZE,
                total_chunks,
            )

            batch = chunks[start:end]

            vectors = embed_with_retry(
                embedder=embedder,
                texts=[
                    chunk.text
                    for chunk in batch
                ],
            )

            payloads = [
                asdict(chunk)
                for chunk in batch
            ]

            point_ids = list(
                range(
                    start + 1,
                    end + 1,
                )
            )

            store.upsert_points(
                point_ids=point_ids,
                vectors=vectors,
                payloads=payloads,
            )

            print(
                f"Indexed {end}/{total_chunks}"
            )

            # Avoid immediately firing the next
            # embedding batch into the free-tier quota.
            if end < total_chunks:
                time.sleep(
                    BATCH_DELAY_SECONDS
                )

        stored_count = store.count()

        if stored_count != total_chunks:
            raise ValueError(
                "Final Qdrant point count mismatch. "
                f"Expected {total_chunks}, "
                f"got {stored_count}."
            )

        print("\nDENSE INDEX SUMMARY")
        print(
            f"Expected chunks: {total_chunks}"
        )
        print(
            f"Stored Qdrant points: {stored_count}"
        )

        print(
            "\nDENSE INDEX BUILD: PASS"
        )

    finally:
        store.close()


if __name__ == "__main__":
    main()