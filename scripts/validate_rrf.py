from src.retrieval.rrf import reciprocal_rank_fusion


def main() -> None:
    dense_results = [
        {
            "chunk_id": "A",
            "rank": 1,
            "score": 0.91,
            "chunk": {"text": "Chunk A"},
        },
        {
            "chunk_id": "C",
            "rank": 2,
            "score": 0.86,
            "chunk": {"text": "Chunk C"},
        },
        {
            "chunk_id": "B",
            "rank": 3,
            "score": 0.82,
            "chunk": {"text": "Chunk B"},
        },
    ]

    sparse_results = [
        {
            "chunk_id": "B",
            "rank": 1,
            "score": 6.20,
            "chunk": {"text": "Chunk B"},
        },
        {
            "chunk_id": "A",
            "rank": 2,
            "score": 5.80,
            "chunk": {"text": "Chunk A"},
        },
        {
            "chunk_id": "D",
            "rank": 3,
            "score": 5.10,
            "chunk": {"text": "Chunk D"},
        },
    ]

    results = reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
    )

    print("\nRRF VALIDATION")
    print("\nFUSED RESULTS")

    for result in results:
        print(
            f"{result['rank']}. "
            f"chunk_id={result['chunk_id']} "
            f"rrf={result['rrf_score']:.6f} "
            f"dense_rank={result['dense_rank']} "
            f"sparse_rank={result['sparse_rank']}"
        )

    assert results[0]["chunk_id"] == "A"
    assert results[1]["chunk_id"] == "B"

    assert (
        results[0]["rrf_score"]
        > results[2]["rrf_score"]
    )

    print("\nRRF: PASS")


if __name__ == "__main__":
    main()