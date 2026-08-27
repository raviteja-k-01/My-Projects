import json
from pathlib import Path
from statistics import mean
from typing import Any

from src.evaluation.answer_judge import (
    AnswerQualityJudge,
)
from src.evaluation.loader import (
    load_gold_questions,
)
from src.rag.rag_pipeline import (
    EnterpriseRAGPipeline,
    DEFAULT_DENSE_LIMIT,
    DEFAULT_SPARSE_LIMIT,
    DEFAULT_FUSED_LIMIT,
    DEFAULT_RERANK_LIMIT,
)


OUTPUT_PATH = Path(
    "data/evaluation/answer_quality_ab_pilot.jsonl"
)


# Small pilot first.
#
# Includes:
# - cases where reranker improved retrieval
# - cases where reranker worsened retrieval
# - one unanswerable question
PILOT_QUESTION_IDS = {
    "Q008",
    "Q012",
    "Q020",
    "Q021",
    "Q026",
}


def generate_answer(
    pipeline: EnterpriseRAGPipeline,
    question: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    context_result = (
        pipeline.context_builder.build(
            candidates=candidates,
        )
    )

    if not context_result["context"]:
        raise ValueError(
            "No usable evidence was found"
        )

    generated = pipeline.generator.generate(
        question=question,
        context=context_result["context"],
    )

    citation_result = (
        pipeline.citation_builder.build(
            answer=generated["answer"],
            evidence=context_result["evidence"],
        )
    )

    return {
        "answer": citation_result["answer"],
        "citations": citation_result[
            "citations"
        ],
        "citation_count": citation_result[
            "citation_count"
        ],
        "evidence_count": context_result[
            "evidence_count"
        ],
        "evidence_tokens": context_result[
            "total_tokens"
        ],
    }


def candidate_ids(
    candidates: list[dict[str, Any]],
) -> list[str]:
    return [
        candidate["chunk_id"]
        for candidate in candidates
    ]


def evaluate_branch(
    judge: AnswerQualityJudge,
    gold: dict[str, Any],
    generated: dict[str, Any],
) -> dict[str, Any]:
    judgment = judge.evaluate(
        question=gold["question"],
        answer=generated["answer"],
        expected_facts=gold[
            "expected_facts"
        ],
        answerable=gold["answerable"],
    )

    return {
        "answer": generated["answer"],
        "citations": generated["citations"],
        "citation_count": generated[
            "citation_count"
        ],
        "evidence_count": generated[
            "evidence_count"
        ],
        "evidence_tokens": generated[
            "evidence_tokens"
        ],
        "judgment": judgment,
    }


def print_branch(
    name: str,
    result: dict[str, Any],
) -> None:
    judgment = result["judgment"]

    print(f"\n{name}")

    print(
        "Fact coverage:",
        judgment[
            "fact_coverage_score"
        ],
    )

    print(
        "Contradiction:",
        judgment[
            "contradiction_found"
        ],
    )

    print(
        "Answerability correct:",
        judgment[
            "answerability_handled_correctly"
        ],
    )

    print(
        "Unsupported/overstated:",
        len(
            judgment[
                "unsupported_or_overstated_claims"
            ]
        ),
    )

    print(
        "Evidence count:",
        result["evidence_count"],
    )


def main() -> None:
    all_gold = load_gold_questions()

    gold_records = [
        record
        for record in all_gold
        if record["question_id"]
        in PILOT_QUESTION_IDS
    ]

    if len(gold_records) != len(
        PILOT_QUESTION_IDS
    ):
        found_ids = {
            record["question_id"]
            for record in gold_records
        }

        missing = (
            PILOT_QUESTION_IDS
            - found_ids
        )

        raise ValueError(
            f"Missing pilot questions: {missing}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pipeline = EnterpriseRAGPipeline()
    judge = AnswerQualityJudge()

    results: list[dict[str, Any]] = []

    print(
        "\nANSWER QUALITY A/B PILOT"
    )

    print(
        "Questions:",
        len(gold_records),
    )

    try:
        for index, gold in enumerate(
            gold_records,
            start=1,
        ):
            question_id = (
                gold["question_id"]
            )

            question = gold["question"]

            print(
                "\n"
                + "=" * 70
            )

            print(
                f"[{index}/{len(gold_records)}] "
                f"{question_id}"
            )

            print(question)

            # -----------------------------------------
            # Retrieve once.
            # Both branches get the same candidate pool.
            # -----------------------------------------

            hybrid_candidates = (
                pipeline.retriever.retrieve(
                    query=question,
                    dense_limit=(
                        DEFAULT_DENSE_LIMIT
                    ),
                    sparse_limit=(
                        DEFAULT_SPARSE_LIMIT
                    ),
                    fused_limit=(
                        DEFAULT_FUSED_LIMIT
                    ),
                )
            )

            # -----------------------------------------
            # A: Hybrid-only top 8
            # -----------------------------------------

            hybrid_top8 = (
                hybrid_candidates[
                    :DEFAULT_RERANK_LIMIT
                ]
            )

            # -----------------------------------------
            # B: Reranked top 8
            # -----------------------------------------

            reranked_top8 = (
                pipeline.reranker.rerank(
                    query=question,
                    candidates=(
                        hybrid_candidates
                    ),
                    limit=(
                        DEFAULT_RERANK_LIMIT
                    ),
                )
            )

            # -----------------------------------------
            # Generate independently
            # -----------------------------------------

            hybrid_generated = (
                generate_answer(
                    pipeline=pipeline,
                    question=question,
                    candidates=hybrid_top8,
                )
            )

            reranked_generated = (
                generate_answer(
                    pipeline=pipeline,
                    question=question,
                    candidates=(
                        reranked_top8
                    ),
                )
            )

            # -----------------------------------------
            # Judge independently
            # -----------------------------------------

            hybrid_result = (
                evaluate_branch(
                    judge=judge,
                    gold=gold,
                    generated=(
                        hybrid_generated
                    ),
                )
            )

            reranked_result = (
                evaluate_branch(
                    judge=judge,
                    gold=gold,
                    generated=(
                        reranked_generated
                    ),
                )
            )

            record = {
                "question_id": (
                    question_id
                ),
                "question": question,
                "question_type": (
                    gold["question_type"]
                ),
                "answerable": (
                    gold["answerable"]
                ),
                "expected_facts": (
                    gold["expected_facts"]
                ),
                "expected_chunk_ids": (
                    gold[
                        "expected_chunk_ids"
                    ]
                ),
                "hybrid_top8_chunk_ids": (
                    candidate_ids(
                        hybrid_top8
                    )
                ),
                "reranked_top8_chunk_ids": (
                    candidate_ids(
                        reranked_top8
                    )
                ),
                "hybrid_only": (
                    hybrid_result
                ),
                "reranked": (
                    reranked_result
                ),
            }

            results.append(record)

            print_branch(
                "A — HYBRID ONLY",
                hybrid_result,
            )

            print_branch(
                "B — RERANKED",
                reranked_result,
            )

            # Save after every question.
            # If a later API request fails,
            # completed cases are not lost.
            with OUTPUT_PATH.open(
                "w",
                encoding="utf-8",
            ) as file:
                for saved in results:
                    file.write(
                        json.dumps(
                            saved,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    finally:
        close_method = getattr(
            pipeline.retriever,
            "close",
            None,
        )

        if callable(close_method):
            close_method()

    # ---------------------------------------------
    # Aggregate comparison
    # ---------------------------------------------

    hybrid_coverages = [
        result["hybrid_only"][
            "judgment"
        ]["fact_coverage_score"]
        for result in results
        if result["hybrid_only"][
            "judgment"
        ]["fact_coverage_score"]
        is not None
    ]

    reranked_coverages = [
        result["reranked"][
            "judgment"
        ]["fact_coverage_score"]
        for result in results
        if result["reranked"][
            "judgment"
        ]["fact_coverage_score"]
        is not None
    ]

    hybrid_contradictions = mean(
        1.0
        if result["hybrid_only"][
            "judgment"
        ]["contradiction_found"]
        else 0.0
        for result in results
    )

    reranked_contradictions = mean(
        1.0
        if result["reranked"][
            "judgment"
        ]["contradiction_found"]
        else 0.0
        for result in results
    )

    hybrid_answerability = mean(
        1.0
        if result["hybrid_only"][
            "judgment"
        ][
            "answerability_handled_correctly"
        ]
        else 0.0
        for result in results
    )

    reranked_answerability = mean(
        1.0
        if result["reranked"][
            "judgment"
        ][
            "answerability_handled_correctly"
        ]
        else 0.0
        for result in results
    )

    hybrid_unsupported_rate = mean(
        1.0
        if result["hybrid_only"][
            "judgment"
        ][
            "unsupported_or_overstated_claims"
        ]
        else 0.0
        for result in results
    )

    reranked_unsupported_rate = mean(
        1.0
        if result["reranked"][
            "judgment"
        ][
            "unsupported_or_overstated_claims"
        ]
        else 0.0
        for result in results
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ANSWER QUALITY A/B PILOT SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        "\nAverage fact coverage"
    )

    print(
        "Hybrid only:",
        f"{mean(hybrid_coverages):.4f}",
    )

    print(
        "Reranked:   ",
        f"{mean(reranked_coverages):.4f}",
    )

    print(
        "\nContradiction rate"
    )

    print(
        "Hybrid only:",
        f"{hybrid_contradictions:.4f}",
    )

    print(
        "Reranked:   ",
        f"{reranked_contradictions:.4f}",
    )

    print(
        "\nAnswerability accuracy"
    )

    print(
        "Hybrid only:",
        f"{hybrid_answerability:.4f}",
    )

    print(
        "Reranked:   ",
        f"{reranked_answerability:.4f}",
    )

    print(
        "\nUnsupported/overstatement rate"
    )

    print(
        "Hybrid only:",
        f"{hybrid_unsupported_rate:.4f}",
    )

    print(
        "Reranked:   ",
        f"{reranked_unsupported_rate:.4f}",
    )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\nANSWER QUALITY A/B PILOT: PASS"
    )


if __name__ == "__main__":
    main()