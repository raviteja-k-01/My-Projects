from src.indexing.bm25_index import BM25Index


EXPECTED_CHUNKS = 1985


def main() -> None:
    print("\nBM25 INDEX BUILD")

    # 1. Build from all retrieval chunks.
    bm25_index = BM25Index()

    chunk_count = bm25_index.build()

    print(
        f"Chunks indexed: {chunk_count}"
    )

    if chunk_count != EXPECTED_CHUNKS:
        raise ValueError(
            "Unexpected BM25 chunk count. "
            f"Expected {EXPECTED_CHUNKS}, "
            f"got {chunk_count}."
        )

    # 2. Persist the BM25 index.
    bm25_index.save()

    print(
        "BM25 index saved: PASS"
    )

    # 3. Load it again from disk.
    loaded_index = BM25Index()

    loaded_index.load()

    print(
        "BM25 index reload: PASS"
    )

    # 4. Test a realistic enterprise query.
    query = (
        "SEC-IR-004 SEV-1 incident reporting"
    )

    results = loaded_index.search(
        query=query,
        limit=5,
    )

    if len(results) != 5:
        raise ValueError(
            "BM25 did not return "
            "the expected number of results"
        )

    print(
        f"\nTEST QUERY: {query}"
    )

    print("\nTOP BM25 RESULTS")

    for result in results:
        chunk = result["chunk"]

        print(
            f"{result['rank']}. "
            f"chunk_id={chunk['chunk_id']} "
            f"document_id={chunk['document_id']} "
            f"score={result['score']:.4f}"
        )

    print(
        "\nBM25 INDEX BUILD: PASS"
    )


if __name__ == "__main__":
    main()