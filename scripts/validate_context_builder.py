from src.retrieval.context_builder import ContextBuilder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker


def main() -> None:
    query = (
        "How should a SEV-1 security incident "
        "be reported?"
    )

    retriever = HybridRetriever()
    reranker = CrossEncoderReranker()

    try:
        candidates = retriever.retrieve(
            query=query,
            dense_limit=20,
            sparse_limit=20,
            fused_limit=20,
        )

        reranked = reranker.rerank(
            query=query,
            candidates=candidates,
            limit=8,
        )

        builder = ContextBuilder(
            max_chunks=8,
            max_context_tokens=4000,
        )

        result = builder.build(
            candidates=reranked,
        )

        print("\nCONTEXT BUILDER VALIDATION")

        print(
            f"\nEvidence count: "
            f"{result['evidence_count']}"
        )

        print(
            f"Total evidence tokens: "
            f"{result['total_tokens']}"
        )

        print("\nEVIDENCE ITEMS")

        for evidence in result["evidence"]:
            print(
                f"\nEvidence {evidence['evidence_id']}"
            )

            print(
                f"Chunk ID: "
                f"{evidence['chunk_id']}"
            )

            print(
                f"Document: "
                f"{evidence['document_id']}"
            )

            print(
                f"Page: "
                f"{evidence['page_start']}"
            )

            print(
                f"Reranker rank: "
                f"{evidence['reranker_rank']}"
            )

            print(
                f"Tokens: "
                f"{evidence['token_count']}"
            )

        print("\nFINAL CONTEXT")
        print(result["context"])

        if result["evidence_count"] > 8:
            raise ValueError(
                "Context contains more than 8 chunks"
            )

        if result["total_tokens"] > 4000:
            raise ValueError(
                "Context exceeded token budget"
            )

        if not result["context"].strip():
            raise ValueError(
                "Context is empty"
            )

        print(
            "\nCONTEXT BUILDER: PASS"
        )

    finally:
        retriever.close()


if __name__ == "__main__":
    main()