from src.generation.answer_generator import GroundedAnswerGenerator


def main() -> None:
    question = (
        "How should a SEV-1 security incident be handled?"
    )

    # Real evidence taken from our validated retrieval/context output.
    context = """
[Evidence 1]
Chunk ID: SEC-IR-004-B0021
Document: SEC-IR-004
Page: 13
Section: 3 | Severity Classification and Reclassification | 3.1 | SEV-1 Critical
Text:
SEV-1 Critical incidents require responders to prioritize
safety, containment, evidence preservation and restoration
of business services. Incident actions, decisions, severity
changes and exceptions must be recorded in the designated
incident record.

[Evidence 2]
Chunk ID: SEC-IR-004-B0044
Document: SEC-IR-004
Page: 26
Section: 6 | Incident Command and Escalation | 6.3 | Executive escalation
Text:
SEV-1 incidents use a named Incident Commander. The Major
Incident Executive Group is activated for SEV-1 incidents
requiring enterprise coordination, client communications,
continuity decisions or executive resources.
""".strip()

    generator = GroundedAnswerGenerator()

    result = generator.generate(
        question=question,
        context=context,
    )

    print("\nANSWER GENERATOR VALIDATION")

    print(
        f"\nModel: {result['model']}"
    )

    print("\nANSWER")
    print(result["answer"])

    print("\nUSAGE")
    print(result["usage"])

    if not result["answer"].strip():
        raise ValueError(
            "Generated answer is empty"
        )

    if "[Evidence" not in result["answer"]:
        raise ValueError(
            "Generated answer contains no evidence citation"
        )

    print(
        "\nANSWER GENERATOR: PASS"
    )


if __name__ == "__main__":
    main()