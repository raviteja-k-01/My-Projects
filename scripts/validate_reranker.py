import time

from src.indexing.bm25_index import BM25Index
from src.retrieval.reranker import CrossEncoderReranker


def main() -> None:
    query = (
        "How should a SEV-1 security incident "
        "be reported?"
    )

    # Load BM25 and get real project candidates.
    bm25 = BM25Index()
    bm25.load()

    candidates = bm25.search(
        query=query,
        limit=20,
    )

    print("\nRERANKER VALIDATION")
    print(f"\nQuery: {query}")
    print(
        f"Candidates before reranking: "
        f"{len(candidates)}"
    )

    # Measure model loading separately.
    load_start = time.perf_counter()

    reranker = CrossEncoderReranker()

    model_load_seconds = (
        time.perf_counter() - load_start
    )

    # Measure actual reranking time.
    rerank_start = time.perf_counter()

    results = reranker.rerank(
        query=query,
        candidates=candidates,
        limit=8,
    )

    rerank_seconds = (
        time.perf_counter() - rerank_start
    )

    if len(results) != 8:
        raise ValueError(
            "Expected 8 reranked results"
        )

    print("\nTOP RERANKED RESULTS")

    for result in results:
        chunk = result["chunk"]

        print(
            f"\nReranker rank: "
            f"{result['reranker_rank']}"
        )

        print(
            f"Original BM25 rank: "
            f"{result['rank']}"
        )

        print(
            f"Chunk ID: "
            f"{chunk['chunk_id']}"
        )

        print(
            f"Document: "
            f"{chunk['document_id']}"
        )

        print(
            f"Reranker score: "
            f"{result['reranker_score']:.4f}"
        )

    print("\nLATENCY")

    print(
        f"Model load time: "
        f"{model_load_seconds:.3f} seconds"
    )

    print(
        f"20-candidate reranking time: "
        f"{rerank_seconds:.3f} seconds"
    )

    print(
        "\nCROSS-ENCODER RERANKER: PASS"
    )


if __name__ == "__main__":
    main()