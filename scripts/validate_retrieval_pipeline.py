import time

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker


def main() -> None:
    query = (
        "How should a SEV-1 security incident "
        "be reported?"
    )

    print("\nRETRIEVAL PIPELINE VALIDATION")
    print(f"\nQuery: {query}")

    retriever = HybridRetriever()
    reranker = CrossEncoderReranker()

    try:
        retrieval_start = time.perf_counter()

        candidates = retriever.retrieve(
            query=query,
            dense_limit=20,
            sparse_limit=20,
            fused_limit=20,
        )

        retrieval_seconds = (
            time.perf_counter()
            - retrieval_start
        )

        print(
            f"\nRRF candidates: "
            f"{len(candidates)}"
        )

        rerank_start = time.perf_counter()

        results = reranker.rerank(
            query=query,
            candidates=candidates,
            limit=8,
        )

        rerank_seconds = (
            time.perf_counter()
            - rerank_start
        )

        if not results:
            raise ValueError(
                "Reranker returned no results"
            )

        print("\nFINAL RERANKED RESULTS")

        for result in results:
            chunk = result["chunk"]
            print(
                f"Text: "
                f"{chunk.get('text', '')[:800]}"
            )
            print(
                f"\nReranker rank: "
                f"{result['reranker_rank']}"
            )

            print(
                f"RRF rank: "
                f"{result['rank']}"
            )

            print(
                f"Chunk ID: "
                f"{result['chunk_id']}"
            )

            print(
                f"Document: "
                f"{chunk.get('document_id')}"
            )

            print(
                f"Dense rank: "
                f"{result['dense_rank']}"
            )

            print(
                f"BM25 rank: "
                f"{result['sparse_rank']}"
            )

            print(
                f"Reranker score: "
                f"{result['reranker_score']:.4f}"
            )

        print("\nLATENCY")

        print(
            f"Hybrid retrieval: "
            f"{retrieval_seconds:.3f} seconds"
        )

        print(
            f"Reranking: "
            f"{rerank_seconds:.3f} seconds"
        )

        print(
            "\nFULL RETRIEVAL PIPELINE: PASS"
        )

    finally:
        retriever.close()


if __name__ == "__main__":
    main()