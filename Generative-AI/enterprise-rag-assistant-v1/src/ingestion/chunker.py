import math
from dataclasses import dataclass

from src.core.models import BodyContentRecord, ChunkRecord
from src.ingestion.text_normalizer import normalize_inline_text

_AVERAGE_CHARACTERS_PER_TOKEN = 4


@dataclass(frozen=True)
class ChunkingConfig:
    """Settings that control section-aware chunk sizes."""

    target_tokens: int = 500
    max_tokens: int = 700
    body_overlap_records: int = 0

    def __post_init__(self) -> None:
        if self.target_tokens <= 0:
            raise ValueError(
                "target_tokens must be greater than zero"
            )

        if self.max_tokens < self.target_tokens:
            raise ValueError(
                "max_tokens must be greater than or equal to "
                "target_tokens"
            )

        if self.body_overlap_records < 0:
            raise ValueError(
                "body_overlap_records cannot be negative"
            )


def estimate_token_count(text: str) -> int:
    """Return a deterministic token estimate for size control."""

    cleaned_text = text.strip()

    if not cleaned_text:
        return 0

    return max(
        1,
        math.ceil(
            len(cleaned_text)
            / _AVERAGE_CHARACTERS_PER_TOKEN
        ),
    )


def format_heading(
    number: str | None,
    title: str | None,
) -> str:
    """Join a heading number and title safely."""

    return " ".join(
        value.strip()
        for value in (number, title)
        if value and value.strip()
    )


def build_heading_context(
    record: BodyContentRecord,
) -> str:
    """Build section and subsection lines for a chunk."""

    section_heading = format_heading(
        record.section_number,
        record.section_title,
    )

    subsection_heading = format_heading(
        record.subsection_number,
        record.subsection_title,
    )

    return "\n".join(
        heading
        for heading in (
            section_heading,
            subsection_heading,
        )
        if heading
    )


def body_context_key(
    record: BodyContentRecord,
) -> tuple[
    str,
    str | None,
    str | None,
    str | None,
    str | None,
    int | None,
]:
    """Return the structural boundary for body chunking."""

    heading_values = (
        record.section_number,
        record.section_title,
        record.subsection_number,
        record.subsection_title,
    )

    page_scope = None

    if not any(heading_values):
        page_scope = record.page_number

    return (
        record.document_id,
        *heading_values,
        page_scope,
    )


def render_body_chunk_text(
    records: list[BodyContentRecord],
) -> str:
    """Render related body records as searchable text."""

    if not records:
        raise ValueError(
            "A body chunk requires at least one record"
        )

    expected_context = body_context_key(records[0])

    for record in records:
        if body_context_key(record) != expected_context:
            raise ValueError(
                "A body chunk cannot mix sections or "
                "subsections"
            )

        if not record.text.strip():
            raise ValueError(
                "A body chunk cannot contain empty records"
            )

    heading_context = build_heading_context(records[0])
    body_parts: list[str] = []

    for index, record in enumerate(records):
        cleaned_text = record.text.strip()

        previous_is_bullet = (
            index > 0
            and records[index - 1].content_type
            == "bullet"
        )
        current_is_bullet = (
            record.content_type == "bullet"
        )

        if previous_is_bullet and current_is_bullet:
            body_parts[-1] = (
                f"{body_parts[-1]}\n{cleaned_text}"
            )
        else:
            body_parts.append(cleaned_text)

    body_text = "\n\n".join(body_parts)

    return "\n\n".join(
        part
        for part in (
            heading_context,
            body_text,
        )
        if part
    )


def create_body_chunk(
    records: list[BodyContentRecord],
    chunk_index: int,
) -> ChunkRecord:
    """Create one body chunk from related records."""

    if chunk_index < 1:
        raise ValueError(
            "chunk_index must start from 1"
        )

    text = render_body_chunk_text(records)
    first_record = records[0]

    for record in records:
        if (
            record.document_name
            != first_record.document_name
        ):
            raise ValueError(
                "A chunk cannot mix document names"
            )

        if (
            record.source_file
            != first_record.source_file
        ):
            raise ValueError(
                "A chunk cannot mix source files"
            )

    page_numbers = [
        record.page_number
        for record in records
    ]

    return ChunkRecord(
        chunk_id=(
            f"{first_record.document_id}"
            f"-B{chunk_index:04d}"
        ),
        chunk_index=chunk_index,
        document_id=first_record.document_id,
        document_name=first_record.document_name,
        source_file=first_record.source_file,
        content_type="body",
        text=text,
        token_count=estimate_token_count(text),
        page_start=min(page_numbers),
        page_end=max(page_numbers),
        section_number=first_record.section_number,
        section_title=first_record.section_title,
        subsection_number=(
            first_record.subsection_number
        ),
        subsection_title=(
            first_record.subsection_title
        ),
    )


def _start_body_group_with_overlap(
    previous_records: list[BodyContentRecord],
    incoming_record: BodyContentRecord,
    config: ChunkingConfig,
) -> list[BodyContentRecord]:
    """Start a size-split group with optional overlap."""

    overlap_count = config.body_overlap_records

    if overlap_count == 0:
        return [incoming_record]

    overlap_records = previous_records[
        -overlap_count:
    ]

    while overlap_records:
        candidate_records = [
            *overlap_records,
            incoming_record,
        ]
        candidate_text = render_body_chunk_text(
            candidate_records
        )
        candidate_tokens = estimate_token_count(
            candidate_text
        )

        if candidate_tokens <= config.max_tokens:
            return candidate_records

        overlap_records = overlap_records[1:]

    return [incoming_record]

def is_retrieval_excluded_record(
    record: BodyContentRecord,
    toc_pages: set[int],
) -> bool:
    """
    Decide whether a preserved body record should be excluded
    from retrieval chunk creation.

    The source record remains in StructuredDocumentRecord.
    """

    normalized_text = normalize_inline_text(
        record.text
    )

    # The whole Table of Contents page is navigational rather
    # than answer-bearing evidence.
    if record.page_number in toc_pages:
        return True

    # This standalone front-matter label has no retrieval value.
    if normalized_text.casefold() == "document control":
        return True

    return False


def filter_retrieval_body_records(
    records: list[BodyContentRecord],
) -> list[BodyContentRecord]:
    """
    Keep source body content intact while selecting only
    retrieval-useful records for Step 6.
    """

    toc_pages = {
        record.page_number
        for record in records
        if (
            normalize_inline_text(
                record.text
            ).casefold()
            == "table of contents"
        )
    }

    return [
        record
        for record in records
        if not is_retrieval_excluded_record(
            record=record,
            toc_pages=toc_pages,
        )
    ]

def chunk_body_content(
    records: list[BodyContentRecord],
    config: ChunkingConfig | None = None,
    starting_index: int = 1,
) -> list[ChunkRecord]:
    """Group ordered body records into section-aware chunks."""

    active_config = config or ChunkingConfig()
    retrieval_records = filter_retrieval_body_records(
        records
    )
    if starting_index < 1:
        raise ValueError(
            "starting_index must start from 1"
        )

    chunks: list[ChunkRecord] = []
    current_records: list[BodyContentRecord] = []
    next_chunk_index = starting_index

    for record in retrieval_records:
        single_record_text = render_body_chunk_text(
            [record]
        )
        single_record_tokens = estimate_token_count(
            single_record_text
        )

        if (
            single_record_tokens
            > active_config.max_tokens
        ):
            raise ValueError(
                "A single body record exceeds "
                "max_tokens: "
                f"{record.document_id}, "
                f"page {record.page_number}"
            )

        if not current_records:
            current_records.append(record)
            continue

        current_context = body_context_key(
            current_records[0]
        )
        incoming_context = body_context_key(record)

        if incoming_context != current_context:
            completed_chunk = create_body_chunk(
                current_records,
                next_chunk_index,
            )
            chunks.append(completed_chunk)

            next_chunk_index += 1
            current_records = [record]
            continue

        current_text = render_body_chunk_text(
            current_records
        )
        current_tokens = estimate_token_count(
            current_text
        )

        candidate_records = [
            *current_records,
            record,
        ]
        candidate_text = render_body_chunk_text(
            candidate_records
        )
        candidate_tokens = estimate_token_count(
            candidate_text
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
            completed_chunk = create_body_chunk(
                current_records,
                next_chunk_index,
            )
            chunks.append(completed_chunk)

            next_chunk_index += 1
            current_records = (
                _start_body_group_with_overlap(
                    previous_records=current_records,
                    incoming_record=record,
                    config=active_config,
                )
            )
            continue

        current_records.append(record)

    # The loop cannot automatically flush its final group.
    if current_records:
        final_chunk = create_body_chunk(
            current_records,
            next_chunk_index,
        )
        chunks.append(final_chunk)

    return chunks