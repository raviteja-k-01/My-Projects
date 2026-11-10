from __future__ import annotations

from collections.abc import Iterable

from src.core.models import DocumentMetadataRecord, TableRecord
from src.ingestion.text_normalizer import join_wrapped_lines


CANONICAL_FIELDS = {
    "document id": "document_id",
    "title": "document_name",
    "document name": "document_name",
    "version": "version",
    "effective date": "effective_date",
    "review date": "review_date",
    "classification": "classification",
    "owner": "owner",
    "approver": "approver",
    "department": "department",
    "applicability": "applicability",
    "status": "status",
}


def _clean_text(value: str | None) -> str:
    if value is None:
        return ""
    return join_wrapped_lines(value.splitlines())


def _field_key(value: str | None) -> str:
    return _clean_text(value).rstrip(":").casefold()


def _metadata_rows(table: TableRecord) -> Iterable[tuple[str, str]]:
    """Yield field/value pairs only from the existing metadata table shape."""

    if not table.rows or len(table.rows[0]) < 2:
        return

    header = [_field_key(cell) for cell in table.rows[0][:2]]
    if header != ["field", "controlled value"]:
        return

    for row in table.rows[1:]:
        if len(row) < 2:
            continue

        field_name = _field_key(row[0])
        value = _clean_text(row[1])
        if field_name and value:
            yield field_name, value


def extract_document_metadata(
    tables: list[TableRecord],
    fallback_document_id: str,
    fallback_document_name: str,
    source_file: str,
) -> DocumentMetadataRecord:
    """
    Extract canonical metadata from already-cleaned front-matter tables.

    Repeated front-matter values use the first non-empty occurrence. The title
    is taken when it appears in the fuller controlled-metadata table; filename
    values are fallbacks only.
    """

    values: dict[str, str] = {}

    for table in sorted(tables, key=lambda item: (item.page_number, item.table_index)):
        for field_name, value in _metadata_rows(table):
            canonical_name = CANONICAL_FIELDS.get(field_name)
            if canonical_name is not None and canonical_name not in values:
                values[canonical_name] = value

    document_id = values.get("document_id", fallback_document_id)
    if document_id != fallback_document_id:
        raise ValueError(
            "Document ID mismatch between filename and controlled metadata: "
            f"{fallback_document_id!r} != {document_id!r}"
        )

    return DocumentMetadataRecord(
        document_id=document_id,
        document_name=values.get("document_name", fallback_document_name),
        source_file=source_file,
        version=values.get("version"),
        effective_date=values.get("effective_date"),
        review_date=values.get("review_date"),
        classification=values.get("classification"),
        owner=values.get("owner"),
        approver=values.get("approver"),
        department=values.get("department"),
        applicability=values.get("applicability"),
        status=values.get("status"),
    )
