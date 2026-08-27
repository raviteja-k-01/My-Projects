from src.evaluation.retrieval_metrics import (
    evaluate_ranked_chunks,
)


def main() -> None:
    retrieved_chunk_ids = [
        "CHUNK_A",
        "CHUNK_B",
        "CHUNK_C",
        "CHUNK_D",
        "CHUNK_E",
    ]

    expected_chunk_ids = [
        "CHUNK_C",
        "CHUNK_E",
    ]

    result = evaluate_ranked_chunks(
        retrieved_chunk_ids=retrieved_chunk_ids,
        expected_chunk_ids=expected_chunk_ids,
    )

    print("\nRETRIEVAL METRICS VALIDATION")

    print(
        "First relevant rank:",
        result["first_relevant_rank"],
    )

    print(
        "MRR:",
        result["mrr"],
    )

    print(
        "Hit@1:",
        result["hit_at_k"]["hit@1"],
    )

    print(
        "Hit@3:",
        result["hit_at_k"]["hit@3"],
    )

    print(
        "Hit@5:",
        result["hit_at_k"]["hit@5"],
    )

    print(
        "Recall@1:",
        result["recall_at_k"]["recall@1"],
    )

    print(
        "Recall@3:",
        result["recall_at_k"]["recall@3"],
    )

    print(
        "Recall@5:",
        result["recall_at_k"]["recall@5"],
    )

    assert (
        result["first_relevant_rank"] == 3
    )

    assert abs(
        result["mrr"] - (1 / 3)
    ) < 1e-9

    assert (
        result["hit_at_k"]["hit@1"] == 0.0
    )

    assert (
        result["hit_at_k"]["hit@3"] == 1.0
    )

    assert (
        result["hit_at_k"]["hit@5"] == 1.0
    )

    assert (
        result["recall_at_k"]["recall@1"]
        == 0.0
    )

    assert (
        result["recall_at_k"]["recall@3"]
        == 0.5
    )

    assert (
        result["recall_at_k"]["recall@5"]
        == 1.0
    )

    print("\nRETRIEVAL METRICS: PASS")


if __name__ == "__main__":
    main()