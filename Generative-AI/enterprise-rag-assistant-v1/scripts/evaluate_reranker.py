import json
from pathlib import Path
from statistics import mean
from typing import Any

from src.evaluation.loader import (
    load_gold_questions,
)
from src.evaluation.retrieval_metrics import (
    evaluate_ranked_chunks,
)
from src.retrieval.hybrid_retriever import (
    HybridRetriever,
)
from src.retrieval.reranker import (
    CrossEncoderReranker,
)


OUTPUT_PATH = Path(
    "data/evaluation/reranker_results.jsonl"
)

K_VALUES = (1, 3, 5, 8)


def get_chunk_ids(
    results: list[dict[str, Any]],
) -> list[str]:
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


def rank_value(
    rank: int | None,
) -> float:
    """
    Convert None into infinity so rank changes
    can be compared safely.
    """

    if rank is None:
        return float("inf")

    return float(rank)


def classify_change(
    before_rank: int | None,
    after_rank: int | None,
) -> str:
    before = rank_value(before_rank)
    after = rank_value(after_rank)

    if after < before:
        return "improved"

    if after > before:
        return "worsened"

    return "unchanged"


def main() -> None:
    gold_records = load_gold_questions()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    retriever = HybridRetriever()
    reranker = CrossEncoderReranker()

    evaluation_results: list[
        dict[str, Any]
    ] = []

    print(
        "\nRERANKER EVALUATION"
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
                        "evaluable": False,
                    }
                )

                continue

            hybrid_results = retriever.retrieve(
                query=question,
                dense_limit=20,
                sparse_limit=20,
                fused_limit=20,
            )

            hybrid_chunk_ids = get_chunk_ids(
                hybrid_results
            )

            # Fair baseline:
            # first 8 results BEFORE reranking.
            baseline_top8_ids = (
                hybrid_chunk_ids[:8]
            )

            reranked_results = reranker.rerank(
                query=question,
                candidates=hybrid_results,
                limit=8,
            )

            reranked_chunk_ids = get_chunk_ids(
                reranked_results
            )

            baseline_metrics = (
                evaluate_ranked_chunks(
                    retrieved_chunk_ids=(
                        baseline_top8_ids
                    ),
                    expected_chunk_ids=(
                        expected_chunk_ids
                    ),
                    k_values=K_VALUES,
                )
            )

            reranked_metrics = (
                evaluate_ranked_chunks(
                    retrieved_chunk_ids=(
                        reranked_chunk_ids
                    ),
                    expected_chunk_ids=(
                        expected_chunk_ids
                    ),
                    k_values=K_VALUES,
                )
            )

            before_rank = baseline_metrics[
                "first_relevant_rank"
            ]

            after_rank = reranked_metrics[
                "first_relevant_rank"
            ]

            change = classify_change(
                before_rank=before_rank,
                after_rank=after_rank,
            )

            result = {
                "question_id": question_id,
                "question": question,
                "question_type": (
                    gold["question_type"]
                ),
                "evaluable": True,
                "expected_chunk_ids": (
                    expected_chunk_ids
                ),
                "hybrid_top20_chunk_ids": (
                    hybrid_chunk_ids
                ),
                "baseline_top8_chunk_ids": (
                    baseline_top8_ids
                ),
                "reranked_top8_chunk_ids": (
                    reranked_chunk_ids
                ),
                "baseline_gold_ranks": (
                    get_gold_ranks(
                        baseline_top8_ids,
                        expected_chunk_ids,
                    )
                ),
                "reranked_gold_ranks": (
                    get_gold_ranks(
                        reranked_chunk_ids,
                        expected_chunk_ids,
                    )
                ),
                "baseline_metrics": (
                    baseline_metrics
                ),
                "reranked_metrics": (
                    reranked_metrics
                ),
                "first_relevant_rank_before": (
                    before_rank
                ),
                "first_relevant_rank_after": (
                    after_rank
                ),
                "rank_change": change,
            }

            evaluation_results.append(
                result
            )

            print(
                f"Before rank: {before_rank}"
            )
            print(
                f"After rank:  {after_rank}"
            )
            print(
                f"Change:      {change}"
            )

            print(
                "Recall@8 before:",
                baseline_metrics[
                    "recall_at_k"
                ]["recall@8"],
            )

            print(
                "Recall@8 after: ",
                reranked_metrics[
                    "recall_at_k"
                ]["recall@8"],
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

    evaluable = [
        result
        for result in evaluation_results
        if result.get(
            "evaluable",
            False,
        )
    ]

    if not evaluable:
        raise ValueError(
            "No reranker-evaluable questions."
        )

    print(
        "\n"
        + "=" * 60
    )
    print(
        "RERANKER SUMMARY"
    )
    print(
        "=" * 60
    )

    print(
        "Evaluable questions:",
        len(evaluable),
    )

    improved = sum(
        result["rank_change"] == "improved"
        for result in evaluable
    )

    worsened = sum(
        result["rank_change"] == "worsened"
        for result in evaluable
    )

    unchanged = sum(
        result["rank_change"] == "unchanged"
        for result in evaluable
    )

    print(
        f"\nImproved:  {improved}"
    )
    print(
        f"Worsened:  {worsened}"
    )
    print(
        f"Unchanged: {unchanged}"
    )

    baseline_mrr = mean(
        result["baseline_metrics"]["mrr"]
        for result in evaluable
    )

    reranked_mrr = mean(
        result["reranked_metrics"]["mrr"]
        for result in evaluable
    )

    print(
        f"\nBaseline MRR: {baseline_mrr:.4f}"
    )
    print(
        f"Reranked MRR: {reranked_mrr:.4f}"
    )
    print(
        f"MRR delta:    "
        f"{reranked_mrr - baseline_mrr:+.4f}"
    )

    for k in K_VALUES:
        baseline_hit = mean(
            result["baseline_metrics"][
                "hit_at_k"
            ][f"hit@{k}"]
            for result in evaluable
        )

        reranked_hit = mean(
            result["reranked_metrics"][
                "hit_at_k"
            ][f"hit@{k}"]
            for result in evaluable
        )

        baseline_recall = mean(
            result["baseline_metrics"][
                "recall_at_k"
            ][f"recall@{k}"]
            for result in evaluable
        )

        reranked_recall = mean(
            result["reranked_metrics"][
                "recall_at_k"
            ][f"recall@{k}"]
            for result in evaluable
        )

        print(
            f"\nHit@{k}:"
        )
        print(
            f"  Before: {baseline_hit:.4f}"
        )
        print(
            f"  After:  {reranked_hit:.4f}"
        )
        print(
            f"  Delta:  "
            f"{reranked_hit - baseline_hit:+.4f}"
        )

        print(
            f"Recall@{k}:"
        )
        print(
            f"  Before: {baseline_recall:.4f}"
        )
        print(
            f"  After:  {reranked_recall:.4f}"
        )
        print(
            f"  Delta:  "
            f"{reranked_recall - baseline_recall:+.4f}"
        )

    print(
        "\nDetailed results saved to:"
    )
    print(
        OUTPUT_PATH
    )

    print(
        "\nRERANKER EVALUATION: PASS"
    )


if __name__ == "__main__":
    main()