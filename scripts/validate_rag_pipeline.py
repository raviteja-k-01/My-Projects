from src.rag.rag_pipeline import EnterpriseRAGPipeline

def main() -> None:
    question = (
        "How should a SEV-1 security incident "
        "be reported and handled?"
    )

    rag = EnterpriseRAGPipeline()

    try:
        result = rag.answer(
            question=question
        )

        print("\nEND-TO-END RAG VALIDATION")

        print(
            f"\nQUESTION:\n"
            f"{result['question']}"
        )

        print(
            f"\nANSWER:\n"
            f"{result['answer']}"
        )

        print("\nCITATIONS")

        for citation in result["citations"]:
            print(
                f"\nEvidence: "
                f"{citation['evidence_id']}"
            )

            print(
                f"Document: "
                f"{citation['document_id']}"
            )

            print(
                f"Page: "
                f"{citation['page_label']}"
            )

            print(
                f"Section: "
                f"{citation['section_label']}"
            )

            print(
                f"Chunk ID: "
                f"{citation['chunk_id']}"
            )

        print("\nPIPELINE STATS")

        print(
            f"Evidence count: "
            f"{result['evidence_count']}"
        )

        print(
            f"Evidence tokens: "
            f"{result['evidence_tokens']}"
        )

        print(
            f"Citation count: "
            f"{result['citation_count']}"
        )

        print(
            f"Model: "
            f"{result['model']}"
        )

        print("\nLATENCY")

        for name, seconds in (
            result["latency"].items()
        ):
            print(
                f"{name}: "
                f"{seconds:.3f} seconds"
            )

        print("\nTOKEN USAGE")
        print(result["usage"])

        if not result["answer"].strip():
            raise ValueError(
                "Final answer is empty"
            )

        if result["evidence_count"] == 0:
            raise ValueError(
                "No evidence was used"
            )

        if result["citation_count"] == 0:
            raise ValueError(
                "Final answer has no citations"
            )

        print(
            "\nEND-TO-END RAG PIPELINE: PASS"
        )

    finally:
        rag.close()


if __name__ == "__main__":
    main()