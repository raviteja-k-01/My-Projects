from src.indexing.chunk_loader import load_indexing_chunks
from src.indexing.embedding_client import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    GeminiEmbeddingClient,
)


def main() -> None:
    # Load our real 1,985 retrieval chunks.
    chunks = load_indexing_chunks()

    # Use only two chunks for this test.
    sample_chunks = chunks[:2]

    embedder = GeminiEmbeddingClient()

    # Convert two real document chunks into vectors.
    document_vectors = embedder.embed_documents(
        [
            chunk.text
            for chunk in sample_chunks
        ]
    )

    # Convert one user-style question into a query vector.
    query_vector = embedder.embed_query(
        "What policy applies to employees?"
    )

    assert len(document_vectors) == 2

    assert all(
        len(vector) == EMBEDDING_DIMENSION
        for vector in document_vectors
    )

    assert (
        len(query_vector)
        == EMBEDDING_DIMENSION
    )

    assert (
        document_vectors[0]
        != document_vectors[1]
    )

    print("\nEMBEDDING CLIENT VALIDATION")

    print(
        f"Model: {EMBEDDING_MODEL}"
    )

    print(
        f"Embedding dimension: "
        f"{EMBEDDING_DIMENSION}"
    )

    print(
        f"Document vectors created: "
        f"{len(document_vectors)}"
    )

    print(
        "Query vector created: PASS"
    )

    print(
        "Vector dimensions: PASS"
    )

    print(
        "\nGEMINI EMBEDDING CLIENT: PASS"
    )


if __name__ == "__main__":
    main()