from src.evaluation.answer_judge import (
    AnswerQualityJudge,
)
from src.evaluation.loader import (
    load_gold_questions,
)


QUESTION_ID = "Q012"


GOOD_ANSWER = """
SEV-1 incidents require a named Incident Commander.
Responders prioritize safety, containment, evidence
preservation, and restoration of business services.

Incident actions, decisions, severity changes, and
exceptions must be recorded.

The Major Incident Executive Group is activated for
SEV-1 events requiring enterprise coordination,
client communications, continuity decisions, or
executive resources.
""".strip()


OVERSTATED_ANSWER = """
SEV-1 incidents require a named Incident Commander.
Responders prioritize safety, containment, evidence
preservation, and restoration of business services.

Incident actions, decisions, severity changes, and
exceptions must be recorded.

The Major Incident Executive Group is activated for
all SEV-1 incidents.
""".strip()


def print_result(
    title: str,
    result: dict,
) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print(
        "Fact coverage:",
        result["fact_coverage_score"],
    )

    print(
        "Flags:",
        result["fact_coverage_flags"],
    )

    print(
        "Contradiction:",
        result["contradiction_found"],
    )

    print(
        "Contradiction explanation:",
        result["contradiction_explanation"],
    )

    print(
        "Unsupported/overstated:",
        result[
            "unsupported_or_overstated_claims"
        ],
    )

    print(
        "Answerability handled:",
        result[
            "answerability_handled_correctly"
        ],
    )

    print(
        "Rationale:",
        result["rationale"],
    )


def main() -> None:
    gold_records = load_gold_questions()

    gold = next(
        record
        for record in gold_records
        if record["question_id"]
        == QUESTION_ID
    )

    judge = AnswerQualityJudge()

    good_result = judge.evaluate(
        question=gold["question"],
        answer=GOOD_ANSWER,
        expected_facts=gold[
            "expected_facts"
        ],
        answerable=gold["answerable"],
    )

    bad_result = judge.evaluate(
        question=gold["question"],
        answer=OVERSTATED_ANSWER,
        expected_facts=gold[
            "expected_facts"
        ],
        answerable=gold["answerable"],
    )

    print_result(
        "GOOD ANSWER",
        good_result,
    )

    print_result(
        "OVERSTATED ANSWER",
        bad_result,
    )

    # ---------------------------------------------
    # Expected behaviour
    # ---------------------------------------------

    assert (
        good_result["fact_coverage_score"]
        == 1.0
    )

    assert (
        good_result["contradiction_found"]
        is False
    )

    assert (
        good_result[
            "answerability_handled_correctly"
        ]
        is True
    )

    assert (
        bad_result["fact_coverage_score"]
        < 1.0
    )

    assert (
        bad_result["contradiction_found"]
        is True
    )

    assert (
        bad_result[
            "unsupported_or_overstated_claims"
        ]
    )

    print(
        "\nANSWER JUDGE: PASS"
    )


if __name__ == "__main__":
    main()