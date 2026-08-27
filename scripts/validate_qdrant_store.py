from pathlib import Path

from src.indexing.chunk_loader import load_indexing_chunks
from src.indexing.embedding_client import GeminiEmbeddingClient
from src.indexing.qdrant_store import QdrantStore


TEST_COLLECTION = "enterprise_rag_chunks_validation"
TEST_QDRANT_PATH = Path("data/qdrant_validation")


def main() -> None:
    # Load our real retrieval chunks.
    chunks = load_indexing_chunks()

    # Only use two chunks for this validation.
    sample_chunks = chunks[:2]

    embedder = GeminiEmbeddingClient()

    store = QdrantStore(
        collection_name=TEST_COLLECTION,
        local_path=TEST_QDRANT_PATH,
    )

    try:
        # 1. Make sure the Qdrant collection exists.
        store.ensure_collection()

        # 2. Embed two real chunks.
        vectors = embedder.embed_documents(
            [
                chunk.text
                for chunk in sample_chunks
            ]
        )

        # 3. Build simple test payloads.
        payloads = [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "content_type": chunk.content_type,
            }
            for chunk in sample_chunks
        ]

        # 4. Store both vectors + payloads.
        store.upsert_points(
            point_ids=[1, 2],
            vectors=vectors,
            payloads=payloads,
        )

        # 5. Confirm exactly two points exist.
        point_count = store.count()

        assert point_count == 2

        # 6. Create a query embedding.
        query_vector = embedder.embed_query(
            sample_chunks[0].text
        )

        # 7. Search Qdrant.
        results = store.query(
            query_vector=query_vector,
            limit=2,
        )

        assert len(results) == 2

        assert all(
            result.payload is not None
            for result in results
        )

        print("\nQDRANT STORE VALIDATION")

        print(
            f"Collection: {TEST_COLLECTION}"
        )

        print(
            f"Stored points: {point_count}"
        )

        print(
            f"Retrieved points: {len(results)}"
        )

        print("\nSEARCH RESULTS")

        for rank, result in enumerate(
            results,
            start=1,
        ):
            print(
                f"{rank}. "
                f"chunk_id="
                f"{result.payload['chunk_id']} "
                f"score={result.score:.4f}"
            )

        print(
            "\nQDRANT STORE: PASS"
        )

    finally:
        store.close()


if __name__ == "__main__":
    main()