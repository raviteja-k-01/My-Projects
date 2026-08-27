from collections import OrderedDict
from collections.abc import Iterable

from src.core.models import (
    BodyContentRecord,
    PageRecord,
    TableRecord,
    TextSpanRecord,
)
from src.ingestion.body_text_filter import filter_body_spans
from src.ingestion.structure_parser import (
    HeadingState,
    classify_heading_line,
    group_spans_by_line,
    update_heading_state,
)
from src.ingestion.text_normalizer import (
    CANONICAL_BULLET,
    join_wrapped_lines,
    normalize_inline_text,
    split_bullet_prefix,
)


def _line_text(
    spans: list[TextSpanRecord],
) -> str:
    """
    Join spans that PyMuPDF reported on the same visual line.
    """

    ordered_spans = sorted(
        spans,
        key=lambda span: span.span_index,
    )

    return normalize_inline_text(
        " ".join(
            span.text
            for span in ordered_spans
        )
    )


def _line_bbox(
    spans: list[TextSpanRecord],
) -> tuple[float, float, float, float]:
    """
    Calculate one bounding box for a complete visual line.
    """

    x0 = min(span.bbox[0] for span in spans)
    y0 = min(span.bbox[1] for span in spans)
    x1 = max(span.bbox[2] for span in spans)
    y1 = max(span.bbox[3] for span in spans)

    return (
        x0,
        y0,
        x1,
        y1,
    )


def _looks_like_cross_block_bullet_continuation(
    bullet_lines: list[str],
    current_line_text: str,
    previous_line_bbox: tuple[float, float, float, float] | None,
    current_line_bbox: tuple[float, float, float, float],
) -> bool:
    """
    Decide whether text in a new PyMuPDF block is actually
    a wrapped continuation of the currently active bullet.
    """

    if previous_line_bbox is None:
        return False

    previous_text = normalize_inline_text(
        " ".join(bullet_lines)
    )

    current_text = normalize_inline_text(
        current_line_text
    )

    if not previous_text:
        return False

    if not current_text:
        return False

    # If the existing bullet clearly ended as a sentence,
    # the next block should not be joined to it.
    if previous_text.endswith((".", "?", "!")):
        return False

    # The confirmed corpus spillovers begin with lowercase
    # continuation text such as:
    # "treatment."
    # "technical response team."
    # "or service criticality."
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
        return False

    previous_x0, previous_y0, _, previous_y1 = (
        previous_line_bbox
    )

    current_x0, current_y0, _, current_y1 = (
        current_line_bbox
    )

    previous_height = max(
        previous_y1 - previous_y0,
        1.0,
    )

    current_height = max(
        current_y1 - current_y0,
        1.0,
    )

    vertical_gap = (
        current_y0 - previous_y1
    )

    max_vertical_gap = max(
        previous_height,
        current_height,
    ) * 1.25

    # New line should follow the previous line,
    # not overlap significantly or be far below it.
    if vertical_gap < -1.0:
        return False

    if vertical_gap > max_vertical_gap:
        return False

    # Wrapped continuation should have approximately
    # the same indentation.
    if current_x0 < previous_x0 - 4.0:
        return False

    if current_x0 > previous_x0 + 36.0:
        return False

    return True

def _append_record(
    records: list[BodyContentRecord],
    page: PageRecord,
    state: HeadingState,
    block_index: int,
    content_type: str,
    lines: list[str],
) -> None:
    text = join_wrapped_lines(lines)

    if not text:
        return

    if content_type == "bullet":
        text = f"{CANONICAL_BULLET} {text}"

    records.append(
        BodyContentRecord(
            document_id=page.document_id,
            document_name=page.document_name,
            page_number=page.page_number,
            content_type=content_type,
            text=text,
            section_number=state.section_number,
            section_title=state.section_title,
            subsection_number=state.subsection_number,
            subsection_title=state.subsection_title,
            source_file=page.source_file,
            block_index=block_index,
        )
    )


def reconstruct_page_body(
    page: PageRecord,
    tables: list[TableRecord],
    heading_state: HeadingState | None = None,
) -> list[BodyContentRecord]:
    """
    Reconstruct one page of body content.

    Section/subsection state may continue across pages.

    Normal paragraphs still respect PyMuPDF block boundaries.
    Bullets are allowed to continue across adjacent blocks only
    when layout and text evidence show that the second block is
    a wrapped continuation.
    """

    state = (
        heading_state
        if heading_state is not None
        else HeadingState()
    )

    body_span_ids = {
        id(span)
        for span in filter_body_spans(
            page=page,
            tables=tables,
        )
    }

    records: list[BodyContentRecord] = []

    active_block_index: int | None = None

    paragraph_lines: list[str] = []
    paragraph_block_index: int | None = None

    bullet_lines: list[str] = []
    bullet_block_index: int | None = None
    bullet_active = False

    previous_body_line_bbox: (
        tuple[float, float, float, float] | None
    ) = None

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        nonlocal paragraph_block_index

        if (
            paragraph_lines
            and paragraph_block_index is not None
        ):
            _append_record(
                records=records,
                page=page,
                state=state,
                block_index=paragraph_block_index,
                content_type="paragraph",
                lines=paragraph_lines,
            )

        paragraph_lines = []
        paragraph_block_index = None

    def flush_bullet() -> None:
        nonlocal bullet_active
        nonlocal bullet_lines
        nonlocal bullet_block_index

        if (
            bullet_lines
            and bullet_block_index is not None
        ):
            _append_record(
                records=records,
                page=page,
                state=state,
                block_index=bullet_block_index,
                content_type="bullet",
                lines=bullet_lines,
            )

        bullet_lines = []
        bullet_block_index = None
        bullet_active = False

    def flush_block() -> None:
        flush_paragraph()
        flush_bullet()

    for line_spans in group_spans_by_line(
        page.spans
    ):
        block_index = line_spans[0].block_index

        block_changed = (
            active_block_index is not None
            and active_block_index != block_index
        )

        # --------------------------------------------------
        # 1. Heading handling
        # --------------------------------------------------

        heading = classify_heading_line(
            line_spans
        )

        if heading is not None:
            flush_block()

            active_block_index = block_index

            previous_body_line_bbox = None

            update_heading_state(
                state=state,
                heading=heading,
            )

            continue

        # --------------------------------------------------
        # 2. Keep only spans classified as body content
        # --------------------------------------------------

        body_line_spans = [
            span
            for span in line_spans
            if id(span) in body_span_ids
        ]

        if not body_line_spans:
            # A removed caption, table, boilerplate line, etc.
            # is a real barrier. Never bridge across it.
            flush_block()

            active_block_index = block_index

            previous_body_line_bbox = None

            continue

        line_text = _line_text(
            body_line_spans
        )

        if not line_text:
            flush_block()

            active_block_index = block_index

            previous_body_line_bbox = None

            continue

        current_line_bbox = _line_bbox(
            body_line_spans
        )

        is_bullet, bullet_text = (
            split_bullet_prefix(line_text)
        )

        # --------------------------------------------------
        # 3. Handle PyMuPDF block change carefully
        # --------------------------------------------------

        keep_existing_bullet = False

        if (
            block_changed
            and bullet_active
            and not is_bullet
        ):
            keep_existing_bullet = (
                _looks_like_cross_block_bullet_continuation(
                    bullet_lines=bullet_lines,
                    current_line_text=line_text,
                    previous_line_bbox=previous_body_line_bbox,
                    current_line_bbox=current_line_bbox,
                )
            )

        if (
            block_changed
            and not keep_existing_bullet
        ):
            flush_block()

        active_block_index = block_index

        # --------------------------------------------------
        # 4. New bullet
        # --------------------------------------------------

        if is_bullet:
            flush_paragraph()
            flush_bullet()

            bullet_active = True
            bullet_block_index = block_index

            bullet_lines = (
                [bullet_text]
                if bullet_text
                else []
            )

            previous_body_line_bbox = (
                current_line_bbox
            )

            continue

        # --------------------------------------------------
        # 5. Normal text
        # --------------------------------------------------

        if bullet_active:
            # This may be either a normal wrapped line in the
            # same block or a validated continuation from an
            # adjacent block.
            bullet_lines.append(
                line_text
            )

        else:
            if paragraph_block_index is None:
                paragraph_block_index = (
                    block_index
                )

            paragraph_lines.append(
                line_text
            )

        previous_body_line_bbox = (
            current_line_bbox
        )

    flush_block()

    return records


def reconstruct_body_content(
    pages: list[PageRecord],
    tables: Iterable[TableRecord],
) -> list[BodyContentRecord]:
    """
    Reconstruct corpus body content with heading state
    isolated per document.
    """

    tables_by_page: OrderedDict[
        tuple[str, int],
        list[TableRecord],
    ] = OrderedDict()

    for table in tables:
        table_key = (
            table.document_id,
            table.page_number,
        )

        tables_by_page.setdefault(
            table_key,
            [],
        ).append(table)

    records: list[BodyContentRecord] = []

    current_document_id: str | None = None

    state = HeadingState()

    for page in pages:
        if page.document_id != current_document_id:
            current_document_id = (
                page.document_id
            )

            state = HeadingState()

        page_tables = tables_by_page.get(
            (
                page.document_id,
                page.page_number,
            ),
            [],
        )

        records.extend(
            reconstruct_page_body(
                page=page,
                tables=page_tables,
                heading_state=state,
            )
        )

    return records