from src.retrieval.hybrid_retriever import HybridRetriever


def main() -> None:
    query = (
        "SEC-IR-004 SEV-1 incident reporting procedure"
    )

    retriever = HybridRetriever()

    try:
        results = retriever.retrieve(
            query=query,
            dense_limit=10,
            sparse_limit=10,
            fused_limit=10,
        )

        if not results:
            raise ValueError(
                "Hybrid retrieval returned no results"
            )

        print("\nHYBRID RETRIEVAL VALIDATION")
        print(f"\nQUERY: {query}")

        print("\nFUSED RESULTS")

        for result in results:
            chunk = result["chunk"]

            print(
                f"\nRank: {result['rank']}"
            )

            print(
                f"Chunk ID: {result['chunk_id']}"
            )

            print(
                f"Document: "
                f"{chunk.get('document_id')}"
            )

            print(
                f"Page: "
                f"{chunk.get('page_start')}"
            )

            print(
                f"Dense rank: "
                f"{result['dense_rank']}"
            )

            print(
                f"Dense score: "
                f"{result['dense_score']}"
            )

            print(
                f"BM25 rank: "
                f"{result['sparse_rank']}"
            )

            print(
                f"BM25 score: "
                f"{result['sparse_score']}"
            )

            print(
                f"RRF score: "
                f"{result['rrf_score']:.6f}"
            )

        # For this specific query, we expect
        # SEC-IR-004 evidence near the top.
        top_five = results[:5]

        if not any(
            result["chunk"].get("document_id")
            == "SEC-IR-004"
            for result in top_five
        ):
            raise ValueError(
                "Expected SEC-IR-004 evidence "
                "was not found in the top 5"
            )

        print(
            "\nHYBRID RETRIEVAL: PASS"
        )

    finally:
        retriever.close()


if __name__ == "__main__":
    main()