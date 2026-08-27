from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date

from src.ingestion.body_text_filter import (
    CONTINUED_SECTION_PATTERN,
    filter_body_spans,
    is_inside_table,
    is_table_caption,
)
from src.ingestion.pipeline import process_documents
from src.ingestion.structure_parser import map_table_heading_states
from src.ingestion.table_search import table_to_searchable_text
from src.ingestion.text_normalizer import normalize_bullet_characters


EXPECTED_DOCUMENTS = 12
EXPECTED_PAGES = 1304
EXPECTED_TABLES = 219

APPENDICES_TITLE = "Appendices, Forms and Checklists"


def compact_text(text: str) -> str:
    """
    Remove formatting-only differences for exact
    content-conservation checks.
    """

    normalized = normalize_bullet_characters(text)

    return "".join(normalized.split())


def validate_content_conservation(document) -> None:
    """
    Verify that body reconstruction does not lose,
    duplicate, or rewrite filtered source content.
    """

    tables_by_page = defaultdict(list)
    records_by_page = defaultdict(list)

    for table in document.tables:
        tables_by_page[table.page_number].append(table)

    for record in document.body_content:
        records_by_page[record.page_number].append(record)

    for page in document.pages:
        page_tables = tables_by_page[page.page_number]

        filtered_spans = filter_body_spans(
            page=page,
            tables=page_tables,
        )

        source_text = "".join(
            span.text
            for span in filtered_spans
        )

        reconstructed_text = "".join(
            record.text
            for record in records_by_page[
                page.page_number
            ]
        )

        assert (
            compact_text(source_text)
            == compact_text(reconstructed_text)
        ), (
            "Body content changed, disappeared, or "
            "was duplicated during reconstruction: "
            f"{page.document_id} "
            f"page {page.page_number}"
        )

        filtered_ids = {
            id(span)
            for span in filtered_spans
        }

        for span in page.spans:
            if (
                is_inside_table(
                    span,
                    page_tables,
                )
                or is_table_caption(
                    span,
                    page_tables,
                )
            ):
                assert id(span) not in filtered_ids, (
                    "Table content or caption leaked "
                    "into body spans: "
                    f"{page.document_id} "
                    f"page {page.page_number}: "
                    f"{span.text!r}"
                )


def validate_document_metadata(document) -> None:
    """
    Verify canonical document metadata produced
    during ingestion.
    """

    metadata = document.metadata

    required_fields = (
        "document_id",
        "document_name",
        "version",
        "effective_date",
        "review_date",
        "classification",
        "owner",
        "approver",
        "department",
        "applicability",
        "status",
    )

    missing = [
        field_name
        for field_name in required_fields
        if not getattr(
            metadata,
            field_name,
        )
    ]

    assert not missing, (
        f"Missing metadata for "
        f"{metadata.document_id}: "
        f"{missing}"
    )

    assert (
        metadata.title
        == metadata.document_name
    )

    date.fromisoformat(
        metadata.effective_date
    )

    date.fromisoformat(
        metadata.review_date
    )

    normalized_fields = (
        metadata.document_name,
        metadata.owner,
        metadata.approver,
        metadata.department,
        metadata.applicability,
        metadata.status,
    )

    assert all(
        value
        == " ".join(value.split())
        for value in normalized_fields
    )

    assert all(
        page.document_name
        == metadata.document_name
        for page in document.pages
    )

    assert (
        "non- employee"
        not in metadata.applicability
    )


def validate_tables(document) -> None:
    """
    Validate cleaned table extraction and deterministic
    searchable table representation.
    """

    keys = [
        (
            table.document_id,
            table.page_number,
            table.table_index,
        )
        for table in document.tables
    ]

    assert len(keys) == len(set(keys)), (
        "Duplicate table keys in "
        f"{document.metadata.document_id}"
    )

    page_body = defaultdict(str)

    for record in document.body_content:
        page_body[
            record.page_number
        ] += (
            " "
            + record.text
        )

    for table in document.tables:
        assert table.rows, (
            "Empty table rows: "
            f"{table.document_id} "
            f"page {table.page_number} "
            f"table {table.table_index}"
        )

        expected_text = (
            table_to_searchable_text(
                table
            )
        )

        assert (
            table.searchable_text
            == expected_text
        )

        assert table.searchable_text

        assert (
            "non- employee"
            not in table.searchable_text
        )

        for row in table.rows:
            for cell in row:
                normalized_cell = (
                    " ".join(
                        cell.split()
                    )
                )

                if not normalized_cell:
                    continue

                assert (
                    normalized_cell
                    in table.searchable_text
                ), (
                    "Cleaned cell missing "
                    "from searchable text: "
                    f"{table.document_id} "
                    f"page {table.page_number} "
                    f"table {table.table_index}"
                )

        if table.caption:
            normalized_caption = (
                " ".join(
                    table.caption.split()
                )
            )

            assert (
                table.searchable_text.count(
                    normalized_caption
                )
                == 1
            )

            normalized_page_body = (
                " ".join(
                    page_body[
                        table.page_number
                    ].split()
                )
            )

            assert (
                normalized_caption
                not in normalized_page_body
            ), (
                "Table caption duplicated "
                "in body: "
                f"{table.document_id} "
                f"page {table.page_number}"
            )


def validate_body_format(document) -> None:
    """
    Validate normalization and body-content record format.
    """

    for record in document.body_content:
        assert record.text

        assert "\n" not in record.text

        assert "\uf0b7" not in record.text

        assert (
            record.text
            == " ".join(
                record.text.split()
            )
        )

        assert (
            record.content_type
            in {
                "paragraph",
                "bullet",
            }
        )

        assert (
            CONTINUED_SECTION_PATTERN.fullmatch(
                record.text
            )
            is None
        ), (
            "Continuation marker leaked "
            "into body: "
            f"{record.document_id} "
            f"page {record.page_number}: "
            f"{record.text!r}"
        )

        if record.content_type == "bullet":
            assert record.text.startswith(
                "• "
            )
        else:
            assert not record.text.startswith(
                "• "
            )


def validate_table_heading_metadata(
    document,
) -> None:
    """
    Verify that every TableRecord permanently stores
    the same active section/subsection context calculated
    from the document structure.
    """

    expected_states = (
        map_table_heading_states(
            pages=document.pages,
            tables=document.tables,
        )
    )

    for table in document.tables:
        table_key = (
            table.document_id,
            table.page_number,
            table.table_index,
        )

        assert table_key in expected_states, (
            "No expected heading state "
            "found for table: "
            f"{table_key}"
        )

        state = expected_states[
            table_key
        ]

        assert (
            table.section_number
            == state.section_number
        ), (
            "Wrong table section number: "
            f"{table_key}"
        )

        assert (
            table.section_title
            == state.section_title
        ), (
            "Wrong table section title: "
            f"{table_key}"
        )

        assert (
            table.subsection_number
            == state.subsection_number
        ), (
            "Wrong table subsection number: "
            f"{table_key}"
        )

        assert (
            table.subsection_title
            == state.subsection_title
        ), (
            "Wrong table subsection title: "
            f"{table_key}"
        )


def validate_iam_page_57(
    documents_by_id,
) -> None:
    """
    Preserve the important IAM page-57 regression:
    the table stays a table, receives the correct
    section/subsection metadata, and Operational
    scenario S07 remains body content.
    """

    iam = documents_by_id[
        "IAM-STD-03"
    ]

    records = [
        record
        for record in iam.body_content
        if record.page_number == 57
    ]

    tables = [
        table
        for table in iam.tables
        if table.page_number == 57
    ]

    body_text = "\n".join(
        record.text
        for record in records
    )

    table_text = "\n".join(
        table.searchable_text
        for table in tables
    )

    caption = (
        "IAM-STD-03-T04 - "
        "Access Approval Matrix"
    )

    scenario_prefix = (
        "Operational scenario S07:"
    )

    assert len(tables) == 1

    iam_table = tables[0]

    # --------------------------------------------------
    # Table structural context
    # --------------------------------------------------

    assert (
        iam_table.section_number
        == "10"
    )

    assert (
        iam_table.section_title
        == "Access Requests, "
        "Approvals and Fulfilment"
    )

    assert (
        iam_table.subsection_number
        == "10.2"
    )

    assert (
        iam_table.subsection_title
        == "Manager approval"
    )

    # --------------------------------------------------
    # Existing body/table regression
    # --------------------------------------------------

    assert len(records) == 14

    assert sum(
        record.content_type
        == "paragraph"
        for record in records
    ) == 6

    assert sum(
        record.content_type
        == "bullet"
        for record in records
    ) == 8

    assert caption not in body_text

    assert (
        table_text.count(caption)
        == 1
    )

    assert scenario_prefix in body_text

    assert (
        scenario_prefix
        not in table_text
    )

    request_paragraph = next(
        record
        for record in records
        if record.text.startswith(
            "Request initiation is governed"
        )
    )

    assert (
        request_paragraph.section_number
        == "10"
    )

    assert (
        request_paragraph.section_title
        == "Access Requests, "
        "Approvals and Fulfilment"
    )

    assert (
        request_paragraph.subsection_number
        == "10.1"
    )

    assert (
        request_paragraph.subsection_title
        == "Request initiation"
    )

    scenario = next(
        record
        for record in records
        if record.text.startswith(
            scenario_prefix
        )
    )

    assert (
        scenario.section_number
        == "10"
    )

    assert (
        scenario.subsection_number
        == "10.2"
    )


def validate_appendix_context(
    documents,
) -> None:
    """
    Ensure appendix content does not inherit the
    final numbered policy section.
    """

    appendix_records = [
        record
        for document in documents
        for record in document.body_content
        if (
            record.section_title
            == APPENDICES_TITLE
        )
    ]

    assert appendix_records, (
        "No appendix body content "
        "received appendix context"
    )

    assert all(
        record.section_number is None
        for record in appendix_records
    )

    assert all(
        record.subsection_number
        for record in appendix_records
    )


def validate_cross_block_bullet_continuations(
    documents,
) -> None:
    """
    Verify that the known PyMuPDF cross-block bullet
    continuations are no longer emitted as standalone
    paragraphs.
    """

    bad_fragments = {
        "treatment.",
        "technical response team.",
        "or service criticality.",
    }

    malformed_records = []

    for document in documents:
        for record in document.body_content:
            normalized_text = (
                record.text
                .strip()
                .lower()
            )

            if (
                record.content_type
                == "paragraph"
                and normalized_text
                in bad_fragments
            ):
                malformed_records.append(
                    record
                )

    print(
        "\nCROSS-BLOCK BULLET CHECK"
    )

    print(
        "Malformed standalone continuations:",
        len(malformed_records),
    )

    for record in malformed_records[:20]:
        print(
            record.document_id,
            f"page {record.page_number}",
            record.content_type,
            repr(record.text),
        )

    assert not malformed_records, (
        "Known cross-block bullet "
        "continuations are still being "
        "created as standalone paragraphs."
    )


def validate_suspicious_short_continuations(
    documents,
) -> None:
    """
    Corpus-level guard against other short lowercase
    paragraphs that may actually be unfinished bullet
    continuations.
    """

    suspicious = []

    for document in documents:
        records = document.body_content

        for previous, current in zip(
            records,
            records[1:],
        ):
            if (
                previous.page_number
                != current.page_number
            ):
                continue

            if (
                previous.content_type
                != "bullet"
            ):
                continue

            if (
                current.content_type
                != "paragraph"
            ):
                continue

            current_text = (
                current.text.strip()
            )

            if not current_text:
                continue

            if (
                len(
                    current_text.split()
                )
                > 5
            ):
                continue

            first_alpha = next(
                (
                    char
                    for char in current_text
                    if char.isalpha()
                ),
                None,
            )

            if (
                first_alpha is None
                or not first_alpha.islower()
            ):
                continue

            suspicious.append(
                (
                    current.document_id,
                    current.page_number,
                    previous.text,
                    current.text,
                )
            )

    print(
        "Suspicious short continuation "
        "paragraphs:",
        len(suspicious),
    )

    for (
        document_id,
        page_number,
        previous_text,
        current_text,
    ) in suspicious[:20]:

        print(
            "\n",
            document_id,
            f"page {page_number}",
        )

        print(
            "Previous bullet:",
            repr(previous_text),
        )

        print(
            "Possible continuation:",
            repr(current_text),
        )

    assert not suspicious, (
        "Suspicious short lowercase "
        "paragraphs remain after bullets. "
        "Inspect the printed examples before "
        "changing reconstruction logic."
    )


def main() -> None:
    documents = process_documents()

    documents_by_id = {
        document.metadata.document_id:
        document
        for document in documents
    }

    total_pages = sum(
        len(document.pages)
        for document in documents
    )

    total_tables = sum(
        len(document.tables)
        for document in documents
    )

    total_body_records = sum(
        len(document.body_content)
        for document in documents
    )

    body_types = Counter(
        record.content_type
        for document in documents
        for record in document.body_content
    )

    # --------------------------------------------------
    # Corpus baseline
    # --------------------------------------------------

    assert (
        len(documents)
        == EXPECTED_DOCUMENTS
    )

    assert (
        len(documents_by_id)
        == EXPECTED_DOCUMENTS
    )

    assert (
        total_pages
        == EXPECTED_PAGES
    )

    assert (
        total_tables
        == EXPECTED_TABLES
    )

    # --------------------------------------------------
    # Steps 1–5 per-document validation
    # --------------------------------------------------

    for document in documents:
        validate_content_conservation(
            document
        )

        validate_document_metadata(
            document
        )

        validate_tables(
            document
        )

        validate_body_format(
            document
        )

        validate_table_heading_metadata(
            document
        )

    # --------------------------------------------------
    # Corpus-level regressions
    # --------------------------------------------------

    validate_iam_page_57(
        documents_by_id
    )

    validate_appendix_context(
        documents
    )

    validate_cross_block_bullet_continuations(
        documents
    )

    validate_suspicious_short_continuations(
        documents
    )

    # --------------------------------------------------
    # Cross-page heading-state check
    # --------------------------------------------------

    heading_pages = {
        (
            document.metadata.document_id,
            heading.page_number,
        )
        for document in documents
        for heading in document.headings
    }

    carried_records = [
        record
        for document in documents
        for record in document.body_content
        if (
            record.section_number
            is not None
            and (
                record.document_id,
                record.page_number,
            )
            not in heading_pages
        )
    ]

    assert carried_records, (
        "No cross-page section state "
        "was retained"
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print(
        "\nINGESTION VALIDATION RESULTS"
    )

    print(
        f"Documents: "
        f"{len(documents)}"
    )

    print(
        f"Pages: "
        f"{total_pages}"
    )

    print(
        f"Tables: "
        f"{total_tables}"
    )

    print(
        f"Body records: "
        f"{total_body_records}"
    )

    print(
        f"Paragraphs: "
        f"{body_types['paragraph']}"
    )

    print(
        f"Bullets: "
        f"{body_types['bullet']}"
    )

    print(
        "Cross-page records with "
        "carried section state: "
        f"{len(carried_records)}"
    )

    print(
        "IAM-STD-03 page 57 "
        "spillover/caption regression: PASS"
    )

    print(
        "Content conservation across "
        "filtered body spans: PASS"
    )

    print(
        "Paragraph reconstruction: PASS"
    )

    print(
        "Body section/subsection metadata: PASS"
    )

    print(
        "Table section/subsection metadata: PASS"
    )

    print(
        "Document metadata extraction: PASS"
    )

    print(
        "Searchable table representation: PASS"
    )

    print(
        "Minimal normalization: PASS"
    )

    print(
        "Cross-block bullet reconstruction: PASS"
    )

    print(
        "\nINGESTION STEPS 1-5: PASS"
    )

    print(
        "\nStep 6 remains PAUSED."
    )

    print(
        "Next checkpoint: "
        "section-aware chunking."
    )


if __name__ == "__main__":
    main()