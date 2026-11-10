import json

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


QUESTION_ID = "Q012"


def generate_answer(
    pipeline: EnterpriseRAGPipeline,
    question: str,
    candidates: list[dict],
) -> dict:
    """
    Build context, generate the answer,
    and build citations for one candidate list.
    """

    context_result = pipeline.context_builder.build(
        candidates=candidates,
    )

    if not context_result["context"]:
        raise ValueError(
            "No usable evidence was found"
        )

    generated = pipeline.generator.generate(
        question=question,
        context=context_result["context"],
    )

    citation_result = pipeline.citation_builder.build(
        answer=generated["answer"],
        evidence=context_result["evidence"],
    )

    return {
        "answer": citation_result["answer"],
        "citations": citation_result["citations"],
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


def print_candidate_ids(
    title: str,
    candidates: list[dict],
) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for rank, candidate in enumerate(
        candidates,
        start=1,
    ):
        print(
            f"{rank:>2}. "
            f"{candidate['chunk_id']}"
        )


def print_answer(
    title: str,
    result: dict,
) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print("\nANSWER:")
    print(result["answer"])

    print("\nCITATIONS:")
    print(
        json.dumps(
            result["citations"],
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )

    print(
        "\nEvidence count:",
        result["evidence_count"],
    )

    print(
        "Evidence tokens:",
        result["evidence_tokens"],
    )


def main() -> None:
    gold_records = load_gold_questions()

    gold = next(
        record
        for record in gold_records
        if record["question_id"]
        == QUESTION_ID
    )

    question = gold["question"]

    print("\nQUESTION:")
    print(question)

    print("\nEXPECTED FACTS:")

    for fact in gold["expected_facts"]:
        print(
            "-",
            fact,
        )

    print("\nGOLD CHUNKS:")

    for chunk_id in gold[
        "expected_chunk_ids"
    ]:
        print(
            "-",
            chunk_id,
        )

    pipeline = EnterpriseRAGPipeline()

    try:
        # -------------------------------------------------
        # Retrieve ONCE.
        #
        # Both A and B start from exactly the same
        # hybrid candidate pool.
        # -------------------------------------------------

        hybrid_candidates = (
            pipeline.retriever.retrieve(
                query=question,
                dense_limit=DEFAULT_DENSE_LIMIT,
                sparse_limit=DEFAULT_SPARSE_LIMIT,
                fused_limit=DEFAULT_FUSED_LIMIT,
            )
        )

        # -------------------------------------------------
        # A: Hybrid-only branch
        # -------------------------------------------------

        hybrid_top8 = hybrid_candidates[
            :DEFAULT_RERANK_LIMIT
        ]

        # -------------------------------------------------
        # B: Reranked branch
        # -------------------------------------------------

        reranked_top8 = (
            pipeline.reranker.rerank(
                query=question,
                candidates=hybrid_candidates,
                limit=DEFAULT_RERANK_LIMIT,
            )
        )

        print_candidate_ids(
            "A — HYBRID-ONLY TOP 8",
            hybrid_top8,
        )

        print_candidate_ids(
            "B — RERANKED TOP 8",
            reranked_top8,
        )

        # -------------------------------------------------
        # Generate answers independently from
        # the two evidence sets.
        # -------------------------------------------------

        hybrid_answer = generate_answer(
            pipeline=pipeline,
            question=question,
            candidates=hybrid_top8,
        )

        reranked_answer = generate_answer(
            pipeline=pipeline,
            question=question,
            candidates=reranked_top8,
        )

        print_answer(
            "A — HYBRID-ONLY ANSWER",
            hybrid_answer,
        )

        print_answer(
            "B — RERANKED ANSWER",
            reranked_answer,
        )

    finally:
        close_method = getattr(
            pipeline.retriever,
            "close",
            None,
        )

        if callable(close_method):
            close_method()


if __name__ == "__main__":
    main()