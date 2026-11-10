from __future__ import annotations

from src.core.models import TableRecord
from src.ingestion.text_normalizer import join_wrapped_lines


def _clean_text(text: str | None) -> str:
    """Collapse layout whitespace without changing the cell's wording."""

    if text is None:
        return ""
    return join_wrapped_lines(text.splitlines())


def _column_labels(header_row: list[str], width: int) -> list[str]:
    labels = []
    used_labels: set[str] = set()

    for column_index in range(width):
        raw_label = header_row[column_index] if column_index < len(header_row) else ""
        label = _clean_text(raw_label) or f"Column {column_index + 1}"

        if label in used_labels:
            label = f"{label} (column {column_index + 1})"

        used_labels.add(label)
        labels.append(label)

    return labels


def table_to_searchable_text(
    table: TableRecord,
    *,
    row_number_start: int = 1,
) -> str:
    """
    Create deterministic, retrieval-friendly text while keeping each row intact.

    The first row is treated as the column header. The original ``rows`` and
    geometric metadata remain unchanged on the TableRecord.
    """
    if row_number_start < 1:
        raise ValueError(
            "row_number_start must start from 1"
        )

    rows = [
        [_clean_text(cell) for cell in row]
        for row in table.rows
    ]

    lines = [
        f"Document ID: {table.document_id}",
        f"Page: {table.page_number}",
        f"Table index: {table.table_index}",
    ]

    if table.caption:
        lines.append(f"Caption: {_clean_text(table.caption)}")

    if not rows:
        return "\n".join(lines)

    width = max(len(row) for row in rows)
    labels = _column_labels(rows[0], width)
    lines.append("Columns: " + " | ".join(labels))

    for row_index, row in enumerate(
        rows[1:],
        start=row_number_start,
    ):
        padded_row = row + [""] * (width - len(row))
        relationships = [
            f"{label}: {value}"
            for label, value in zip(labels, padded_row)
        ]
        lines.append(f"Row {row_index}: " + " | ".join(relationships))

    return "\n".join(lines)
