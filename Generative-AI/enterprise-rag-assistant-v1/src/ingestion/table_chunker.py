from dataclasses import replace

from src.core.models import (
    ChunkRecord,
    DocumentMetadataRecord,
    StructuredDocumentRecord,
    TableRecord,
)
from src.ingestion.chunker import (
    ChunkingConfig,
    estimate_token_count,
    format_heading,
)
from src.ingestion.table_search import (
    table_to_searchable_text,
)
from src.ingestion.text_normalizer import normalize_inline_text


def build_table_heading_context(
    table: TableRecord,
) -> str:
    """
    Build section/subsection context directly from the
    metadata already persisted on TableRecord.
    """

    section_heading = format_heading(
        table.section_number,
        table.section_title,
    )

    subsection_heading = format_heading(
        table.subsection_number,
        table.subsection_title,
    )

    return "\n".join(
        heading
        for heading in (
            section_heading,
            subsection_heading,
        )
        if heading
    )


def render_table_chunk_text(
    table: TableRecord,
    row_start: int | None = None,
    row_end: int | None = None,
) -> str:
    """
    Render either the complete table or a selected
    original row range as retrieval-ready text.
    """

    only_one_boundary_supplied = (
        (row_start is None)
        != (row_end is None)
    )

    if only_one_boundary_supplied:
        raise ValueError(
            "row_start and row_end "
            "must be supplied together"
        )

    if row_start is None:
        # Rebuild from the current rows and caption instead
        # of trusting possibly stale cached searchable_text.
        table_text = (
            table_to_searchable_text(
                table
            )
        )

    else:
        if row_start < 1:
            raise ValueError(
                "table data rows "
                "must start from 1"
            )

        if row_end < row_start:
            raise ValueError(
                "row_end cannot be smaller "
                "than row_start"
            )

        data_row_count = max(
            0,
            len(table.rows) - 1,
        )

        if row_end > data_row_count:
            raise ValueError(
                "table row range exceeds "
                "the original rows"
            )

        # Row 0 is treated as the table header.
        # Keep it in every split table chunk.
        selected_rows = [
            table.rows[0],
            *table.rows[
                row_start:row_end + 1
            ],
        ]

        partial_table = replace(
            table,
            rows=selected_rows,
            searchable_text="",
        )

        table_text = (
            table_to_searchable_text(
                partial_table,
                row_number_start=row_start,
            )
        )

    heading_context = (
        build_table_heading_context(
            table
        )
    )

    # Some appendix table captions also act as their
    # subsection heading. In that case do not repeat
    # the same text twice inside the retrieval chunk.
    subsection_heading = format_heading(
        table.subsection_number,
        table.subsection_title,
    )

    if (
        table.caption
        and subsection_heading
        and normalize_inline_text(
            table.caption
        )
        == normalize_inline_text(
            subsection_heading
        )
    ):
        heading_context = format_heading(
            table.section_number,
            table.section_title,
        )

    return "\n\n".join(
        part
        for part in (
            heading_context,
            table_text,
        )
        if part
    )


def create_table_chunk(
    table: TableRecord,
    metadata: DocumentMetadataRecord,
    chunk_index: int,
    part_index: int,
    row_start: int | None = None,
    row_end: int | None = None,
) -> ChunkRecord:
    """
    Create one retrieval chunk from a complete table
    or one deterministic table-row range.
    """

    if chunk_index < 1:
        raise ValueError(
            "chunk_index must start from 1"
        )

    if part_index < 1:
        raise ValueError(
            "part_index must start from 1"
        )

    if (
        table.document_id
        != metadata.document_id
    ):
        raise ValueError(
            "Table and document metadata IDs "
            "do not match"
        )

    text = render_table_chunk_text(
        table=table,
        row_start=row_start,
        row_end=row_end,
    )

    return ChunkRecord(
        chunk_id=(
            f"{table.document_id}"
            f"-TABLE-P{table.page_number:04d}"
            f"-I{table.table_index:03d}"
            f"-C{part_index:03d}"
        ),
        chunk_index=chunk_index,
        document_id=table.document_id,
        document_name=metadata.document_name,
        source_file=metadata.source_file,
        content_type="table",
        text=text,
        token_count=estimate_token_count(
            text
        ),
        page_start=table.page_number,
        page_end=table.page_number,
        section_number=(
            table.section_number
        ),
        section_title=(
            table.section_title
        ),
        subsection_number=(
            table.subsection_number
        ),
        subsection_title=(
            table.subsection_title
        ),
        table_index=(
            table.table_index
        ),
        table_caption=(
            table.caption
        ),
        table_row_start=row_start,
        table_row_end=row_end,
    )


def chunk_table(
    table: TableRecord,
    metadata: DocumentMetadataRecord,
    config: ChunkingConfig | None = None,
    starting_index: int = 1,
) -> list[ChunkRecord]:
    """
    Split one table into deterministic row-group chunks.

    The whole table is kept together whenever it fits
    inside max_tokens. Oversized tables are split only
    between data rows while repeating their header.
    """

    active_config = (
        config
        or ChunkingConfig()
    )

    if starting_index < 1:
        raise ValueError(
            "starting_index must start from 1"
        )

    data_row_count = max(
        0,
        len(table.rows) - 1,
    )

    # Header-only or empty-data tables still retain
    # caption, heading context and provenance.
    if data_row_count == 0:
        only_chunk = create_table_chunk(
            table=table,
            metadata=metadata,
            chunk_index=starting_index,
            part_index=1,
        )

        if (
            only_chunk.token_count
            > active_config.max_tokens
        ):
            raise ValueError(
                "A table without data rows "
                "exceeds max_tokens: "
                f"{table.document_id}, "
                f"page {table.page_number}, "
                f"table {table.table_index}"
            )

        return [
            only_chunk
        ]

    # Try keeping the entire table together first.
    complete_chunk = create_table_chunk(
        table=table,
        metadata=metadata,
        chunk_index=starting_index,
        part_index=1,
        row_start=1,
        row_end=data_row_count,
    )

    if (
        complete_chunk.token_count
        <= active_config.max_tokens
    ):
        return [
            complete_chunk
        ]

    chunks: list[ChunkRecord] = []

    current_row_start = 1
    current_row_end = 0

    next_chunk_index = (
        starting_index
    )

    next_part_index = 1

    for row_number in range(
        1,
        data_row_count + 1,
    ):
        # Verify that one individual row is not
        # itself larger than the hard chunk limit.
        single_row_text = (
            render_table_chunk_text(
                table=table,
                row_start=row_number,
                row_end=row_number,
            )
        )

        single_row_tokens = (
            estimate_token_count(
                single_row_text
            )
        )

        if (
            single_row_tokens
            > active_config.max_tokens
        ):
            raise ValueError(
                "A single table row "
                "exceeds max_tokens: "
                f"{table.document_id}, "
                f"page {table.page_number}, "
                f"table {table.table_index}, "
                f"row {row_number}"
            )

        # Start the first row group.
        if (
            current_row_end
            < current_row_start
        ):
            current_row_end = (
                row_number
            )

            continue

        current_text = (
            render_table_chunk_text(
                table=table,
                row_start=current_row_start,
                row_end=current_row_end,
            )
        )

        current_tokens = (
            estimate_token_count(
                current_text
            )
        )

        candidate_text = (
            render_table_chunk_text(
                table=table,
                row_start=current_row_start,
                row_end=row_number,
            )
        )

        candidate_tokens = (
            estimate_token_count(
                candidate_text
            )
        )

        target_already_reached = (
            current_tokens
            >= active_config.target_tokens
        )

        maximum_would_be_exceeded = (
            candidate_tokens
            > active_config.max_tokens
        )

        if (
            target_already_reached
            or maximum_would_be_exceeded
        ):
            completed_chunk = (
                create_table_chunk(
                    table=table,
                    metadata=metadata,
                    chunk_index=(
                        next_chunk_index
                    ),
                    part_index=(
                        next_part_index
                    ),
                    row_start=(
                        current_row_start
                    ),
                    row_end=(
                        current_row_end
                    ),
                )
            )

            chunks.append(
                completed_chunk
            )

            next_chunk_index += 1
            next_part_index += 1

            current_row_start = (
                row_number
            )

            current_row_end = (
                row_number
            )

            continue

        current_row_end = (
            row_number
        )

    final_chunk = create_table_chunk(
        table=table,
        metadata=metadata,
        chunk_index=next_chunk_index,
        part_index=next_part_index,
        row_start=current_row_start,
        row_end=current_row_end,
    )

    chunks.append(
        final_chunk
    )

    return chunks


def chunk_document_tables(
    document: StructuredDocumentRecord,
    config: ChunkingConfig | None = None,
    starting_index: int = 1,
) -> list[ChunkRecord]:
    """
    Chunk every table in one document in source order.

    Structural metadata is consumed directly from the
    already-complete TableRecord objects produced by
    ingestion Steps 1–5.
    """

    active_config = (
        config
        or ChunkingConfig()
    )

    if starting_index < 1:
        raise ValueError(
            "starting_index must start from 1"
        )

    chunks: list[ChunkRecord] = []

    next_chunk_index = (
        starting_index
    )

    ordered_tables = sorted(
        document.tables,
        key=lambda table: (
            table.page_number,
            table.bbox[1],
            table.table_index,
        ),
    )

    for table in ordered_tables:
        table_chunks = chunk_table(
            table=table,
            metadata=document.metadata,
            config=active_config,
            starting_index=(
                next_chunk_index
            ),
        )

        chunks.extend(
            table_chunks
        )

        next_chunk_index += len(
            table_chunks
        )

    return chunks