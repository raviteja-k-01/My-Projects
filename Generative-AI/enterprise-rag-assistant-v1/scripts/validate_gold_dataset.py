from collections import Counter

from src.evaluation.loader import (
    load_gold_questions,
)


def main() -> None:
    records = load_gold_questions()

    print("\nGOLD DATASET VALIDATION")
    print(f"Total questions: {len(records)}")

    if len(records) != 30:
        raise ValueError(
            f"Expected 30 questions, "
            f"found {len(records)}"
        )

    question_types = Counter(
        record["question_type"]
        for record in records
    )

    answerable_count = sum(
        1
        for record in records
        if record["answerable"]
    )

    unanswerable_count = (
        len(records) - answerable_count
    )

    print("\nQUESTION TYPES")

    for question_type, count in sorted(
        question_types.items()
    ):
        print(
            f"{question_type}: {count}"
        )

    print("\nANSWERABILITY")
    print(
        f"Answerable: {answerable_count}"
    )
    print(
        f"Unanswerable: {unanswerable_count}"
    )

    print("\nFIRST RECORD")
    print(
        f"ID: {records[0]['question_id']}"
    )
    print(
        f"Question: {records[0]['question']}"
    )
    print(
        "Expected chunks:",
        records[0]["expected_chunk_ids"],
    )

    print("\nGOLD DATASET: PASS")


if __name__ == "__main__":
    main()