import json
from pathlib import Path
from statistics import mean
from typing import Any

from src.evaluation.loader import (
    load_gold_questions,
)
from src.evaluation.retrieval_metrics import (
    DEFAULT_K_VALUES,
    evaluate_ranked_chunks,
)
from src.retrieval.hybrid_retriever import (
    HybridRetriever,
)


OUTPUT_PATH = Path(
    "data/evaluation/hybrid_retrieval_results.jsonl"
)


def get_chunk_ids(
    results: list[dict[str, Any]],
) -> list[str]:
    """
    Extract chunk IDs from HybridRetriever results.
    """

    chunk_ids: list[str] = []

    for result in results:
        chunk_id = result.get("chunk_id")

        if isinstance(chunk_id, str):
            chunk_ids.append(chunk_id)

    return chunk_ids


def get_gold_ranks(
    retrieved_chunk_ids: list[str],
    expected_chunk_ids: list[str],
) -> dict[str, int | None]:
    """
    Show the rank at which every gold chunk appeared.
    """

    rank_lookup = {
        chunk_id: rank
        for rank, chunk_id in enumerate(
            retrieved_chunk_ids,
            start=1,
        )
    }

    return {
        chunk_id: rank_lookup.get(chunk_id)
        for chunk_id in expected_chunk_ids
    }


def main() -> None:
    gold_records = load_gold_questions()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    retriever = HybridRetriever()

    evaluation_results: list[dict[str, Any]] = []

    print(
        "\nHYBRID RETRIEVAL EVALUATION"
    )
    print(
        f"Gold questions: {len(gold_records)}"
    )

    try:
        for index, gold in enumerate(
            gold_records,
            start=1,
        ):
            question_id = gold["question_id"]
            question = gold["question"]
            expected_chunk_ids = (
                gold["expected_chunk_ids"]
            )

            print(
                f"\n[{index}/{len(gold_records)}] "
                f"{question_id}"
            )
            print(
                f"Question: {question}"
            )

            # Some unanswerable questions may have
            # no expected retrieval chunk.
            if not expected_chunk_ids:
                print(
                    "SKIPPED: no gold retrieval chunks"
                )

                evaluation_results.append(
                    {
                        "question_id": question_id,
                        "question": question,
                        "question_type": (
                            gold["question_type"]
                        ),
                        "expected_chunk_ids": [],
                        "retrieved_chunk_ids": [],
                        "gold_ranks": {},
                        "metrics": {
                            "evaluable": False,
                        },
                    }
                )

                continue

            results = retriever.retrieve(
                query=question,
                dense_limit=20,
                sparse_limit=20,
                fused_limit=20,
            )

            retrieved_chunk_ids = get_chunk_ids(
                results
            )

            metrics = evaluate_ranked_chunks(
                retrieved_chunk_ids=(
                    retrieved_chunk_ids
                ),
                expected_chunk_ids=(
                    expected_chunk_ids
                ),
            )

            gold_ranks = get_gold_ranks(
                retrieved_chunk_ids=(
                    retrieved_chunk_ids
                ),
                expected_chunk_ids=(
                    expected_chunk_ids
                ),
            )

            evaluation_results.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "question_type": (
                        gold["question_type"]
                    ),
                    "expected_chunk_ids": (
                        expected_chunk_ids
                    ),
                    "retrieved_chunk_ids": (
                        retrieved_chunk_ids
                    ),
                    "gold_ranks": gold_ranks,
                    "metrics": metrics,
                }
            )

            print(
                "Gold ranks:",
                gold_ranks,
            )

            print(
                "First relevant rank:",
                metrics[
                    "first_relevant_rank"
                ],
            )

            print(
                "Hit@5:",
                metrics[
                    "hit_at_k"
                ]["hit@5"],
            )

            print(
                "Recall@20:",
                metrics[
                    "recall_at_k"
                ]["recall@20"],
            )

    finally:
        close_method = getattr(
            retriever,
            "close",
            None,
        )

        if callable(close_method):
            close_method()

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        for result in evaluation_results:
            file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    evaluable_results = [
        result
        for result in evaluation_results
        if result["metrics"].get(
            "evaluable",
            False,
        )
    ]

    if not evaluable_results:
        raise ValueError(
            "No retrieval-evaluable questions found."
        )

    print(
        "\n"
        + "=" * 60
    )
    print(
        "HYBRID RETRIEVAL SUMMARY"
    )
    print(
        "=" * 60
    )

    print(
        "Evaluable questions:",
        len(evaluable_results),
    )

    print(
        "Skipped questions:",
        len(evaluation_results)
        - len(evaluable_results),
    )

    mrr = mean(
        result["metrics"]["mrr"]
        for result in evaluable_results
    )

    print(
        f"\nMRR: {mrr:.4f}"
    )

    for k in DEFAULT_K_VALUES:
        hit = mean(
            result["metrics"][
                "hit_at_k"
            ][f"hit@{k}"]
            for result in evaluable_results
        )

        recall = mean(
            result["metrics"][
                "recall_at_k"
            ][f"recall@{k}"]
            for result in evaluable_results
        )

        print(
            f"Hit@{k}:    {hit:.4f}"
        )
        print(
            f"Recall@{k}: {recall:.4f}"
        )

    print(
        "\nDetailed results saved to:"
    )
    print(
        OUTPUT_PATH
    )

    print(
        "\nHYBRID RETRIEVAL EVALUATION: PASS"
    )


if __name__ == "__main__":
    main()