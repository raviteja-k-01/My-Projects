import json
from pathlib import Path
from typing import Any


DEFAULT_GOLD_PATH = Path(
    "data/evaluation/gold_questions.jsonl"
)


REQUIRED_FIELDS = {
    "question_id",
    "question",
    "question_type",
    "expected_document_id",
    "expected_section",
    "expected_chunk_ids",
    "expected_facts",
    "answerable",
}


def load_gold_questions(
    path: str | Path = DEFAULT_GOLD_PATH,
) -> list[dict[str, Any]]:
    """
    Load and validate the gold RAG evaluation dataset.

    Each JSONL line represents one evaluation question
    together with its expected evidence and facts.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Gold evaluation file not found: {path}"
        )

    records: list[dict[str, Any]] = []
    seen_question_ids: set[str] = set()

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line "
                    f"{line_number}: {exc}"
                ) from exc

            missing_fields = (
                REQUIRED_FIELDS - record.keys()
            )

            if missing_fields:
                raise ValueError(
                    f"Line {line_number} is missing "
                    f"required fields: "
                    f"{sorted(missing_fields)}"
                )

            question_id = record["question_id"]

            if not isinstance(
                question_id,
                str,
            ) or not question_id.strip():
                raise ValueError(
                    f"Invalid question_id on "
                    f"line {line_number}"
                )

            if question_id in seen_question_ids:
                raise ValueError(
                    f"Duplicate question_id: "
                    f"{question_id}"
                )

            seen_question_ids.add(
                question_id
            )

            question = record["question"]

            if not isinstance(
                question,
                str,
            ) or not question.strip():
                raise ValueError(
                    f"Question is empty for "
                    f"{question_id}"
                )

            if not isinstance(
                record["expected_chunk_ids"],
                list,
            ):
                raise ValueError(
                    f"expected_chunk_ids must be "
                    f"a list for {question_id}"
                )

            if not isinstance(
                record["expected_facts"],
                list,
            ):
                raise ValueError(
                    f"expected_facts must be "
                    f"a list for {question_id}"
                )

            if not isinstance(
                record["answerable"],
                bool,
            ):
                raise ValueError(
                    f"answerable must be boolean "
                    f"for {question_id}"
                )

            records.append(
                record
            )

    if not records:
        raise ValueError(
            "Gold evaluation dataset is empty"
        )

    return records