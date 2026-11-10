import re

from src.core.models import (
    PageRecord,
    TableRecord,
    TextSpanRecord,
)
from src.ingestion.structure_parser import (
    classify_heading_line,
    group_spans_by_line,
)
from src.ingestion.text_normalizer import normalize_inline_text


CONTINUED_SECTION_PATTERN = re.compile(
    r"^(?:\d+\s+.+|Table of Contents|Appendices)\s+-\s+continued$",
    re.IGNORECASE,
)


def span_inside_bbox(
    span_bbox: tuple[float, float, float, float],
    region_bbox: tuple[float, float, float, float],
) -> bool:
    sx0, sy0, sx1, sy1 = span_bbox
    rx0, ry0, rx1, ry1 = region_bbox

    center_x = (sx0 + sx1) / 2
    center_y = (sy0 + sy1) / 2

    return (
        rx0 <= center_x <= rx1
        and ry0 <= center_y <= ry1
    )


def is_inside_table(
    span: TextSpanRecord,
    tables: list[TableRecord],
) -> bool:
    for table in tables:
        if span_inside_bbox(
            span_bbox=span.bbox,
            region_bbox=table.bbox,
        ):
            return True

    return False


def is_page_boilerplate(
    span: TextSpanRecord,
    page: PageRecord,
) -> bool:
    _, y0, _, y1 = span.bbox

    top_limit = page.page_height * 0.05
    bottom_limit = page.page_height * 0.95

    # Bold carry-forward marker printed at the top of continuation pages. The
    # active section is already retained by the heading state.
    if (
        y0 <= page.page_height * 0.08
        and "Bold" in span.font_name
        and CONTINUED_SECTION_PATTERN.fullmatch(normalize_inline_text(span.text))
    ):
        return True

    # Repeated small header/footer text.
    if span.font_size <= 9.5:
        if y1 <= top_limit or y0 >= bottom_limit:
            return True

    # Repeated italic metadata line.
    if (
        "Italic" in span.font_name
        and span.font_size <= 8.0
        and "|" in span.text
    ):
        return True

    return False


def normalize_text(text: str) -> str:
    """
    Normalize whitespace only for comparison.
    Does not modify the actual stored body text.
    """
    return normalize_inline_text(text)


def is_table_caption(
    span: TextSpanRecord,
    tables: list[TableRecord],
) -> bool:
    span_text = normalize_text(span.text)

    for table in tables:
        if table.caption is None:
            continue

        caption_text = normalize_text(table.caption)

        if span_text == caption_text:
            return True

    return False


def filter_body_spans(
    page: PageRecord,
    tables: list[TableRecord],
) -> list[TextSpanRecord]:

    body_spans = []
    heading_span_ids = {
        id(span)
        for line_spans in group_spans_by_line(page.spans)
        if classify_heading_line(line_spans) is not None
        for span in line_spans
    }

    for span in page.spans:

        # 1. Section/subsection headings are handled separately.
        if id(span) in heading_span_ids:
            continue

        # 2. Remove repeated page boilerplate.
        if is_page_boilerplate(
            span=span,
            page=page,
        ):
            continue

        # 3. Remove table caption from body.
        # It is already preserved in TableRecord.caption.
        if is_table_caption(
            span=span,
            tables=tables,
        ):
            continue

        # 4. Remove actual table cells from body.
        # They are already preserved in TableRecord.rows.
        if is_inside_table(
            span=span,
            tables=tables,
        ):
            continue

        # 5. Everything remaining is body content.
        body_spans.append(span)

    return body_spans
