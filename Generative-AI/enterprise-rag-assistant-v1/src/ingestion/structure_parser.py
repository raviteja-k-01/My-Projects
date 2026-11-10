from collections import OrderedDict
from dataclasses import dataclass
import re

from src.core.models import (
    HeadingRecord,
    PageRecord,
    TableRecord,
    TextSpanRecord,
)
from src.ingestion.text_normalizer import normalize_inline_text


MAIN_SECTION_PATTERN = re.compile(
    r"^(\d+)\s+(.+)$"
)

SUBSECTION_PATTERN = re.compile(
    r"^(\d+\.\d+)\s+(.+)$"
)

APPENDICES_SECTION_PATTERN = re.compile(
    r"^Appendices,\s+Forms\s+and\s+Checklists$",
    re.IGNORECASE,
)

APPENDIX_ITEM_PATTERN = re.compile(
    r"^((?:[A-Z0-9]+-){3,}(?:A[A-Z]|F\d{2}))\s+(.+)$",
    re.IGNORECASE,
)


@dataclass
class HeadingState:
    section_number: str | None = None
    section_title: str | None = None
    subsection_number: str | None = None
    subsection_title: str | None = None


def classify_heading(
    text: str,
    font_size: float,
    font_name: str,
):
    """
    Decide whether a span is a main section,
    subsection, or normal text.
    """

    is_bold = "Bold" in font_name

    if not is_bold:
        return None

    if (
        APPENDICES_SECTION_PATTERN.fullmatch(text)
        and 14.5 <= font_size <= 15.5
    ):
        return {
            "type": "section",
            "number": "",
            "title": text,
        }

    appendix_item_match = APPENDIX_ITEM_PATTERN.match(text)

    if (
        appendix_item_match
        and 10.5 <= font_size <= 11.5
    ):
        return {
            "type": "subsection",
            "number": appendix_item_match.group(1),
            "title": appendix_item_match.group(2),
        }

    subsection_match = SUBSECTION_PATTERN.match(text)

    if (
        subsection_match
        and 10.5 <= font_size <= 11.5
    ):
        return {
            "type": "subsection",
            "number": subsection_match.group(1),
            "title": subsection_match.group(2),
        }

    section_match = MAIN_SECTION_PATTERN.match(text)

    if (
        section_match
        and 14.5 <= font_size <= 15.5
    ):
        return {
            "type": "section",
            "number": section_match.group(1),
            "title": section_match.group(2),
        }

    return None


def group_spans_by_line(
    spans: list[TextSpanRecord],
) -> list[list[TextSpanRecord]]:
    """Keep PyMuPDF's block/line reading order while grouping line spans."""

    grouped: OrderedDict[tuple[int, int], list[TextSpanRecord]] = OrderedDict()

    for span in spans:
        line_key = (span.block_index, span.line_index)
        grouped.setdefault(line_key, []).append(span)

    return [
        sorted(line_spans, key=lambda span: span.span_index)
        for line_spans in grouped.values()
    ]


def classify_heading_line(
    spans: list[TextSpanRecord],
):
    """Classify a heading even when its number/title use separate spans."""

    if (
        len(spans) >= 2
        and all("Bold" in span.font_name for span in spans)
    ):
        combined_text = normalize_inline_text(
            " ".join(span.text for span in spans)
        )
        representative_size = sum(
            span.font_size
            for span in spans
        ) / len(spans)

        heading = classify_heading(
            text=combined_text,
            font_size=representative_size,
            font_name="Bold",
        )

        if heading is not None:
            return heading

    for span in spans:
        heading = classify_heading(
            text=span.text,
            font_size=span.font_size,
            font_name=span.font_name,
        )

        if heading is not None:
            return heading

    return None


def update_heading_state(
    state: HeadingState,
    heading: dict[str, str],
) -> HeadingState:
    """Apply one classified heading to the active document state."""

    if heading["type"] == "section":
        state.section_number = heading["number"] or None
        state.section_title = heading["title"]

        # A new main section closes the previous subsection.
        state.subsection_number = None
        state.subsection_title = None

    elif heading["type"] == "subsection":
        state.subsection_number = heading["number"]
        state.subsection_title = heading["title"]

    return state


def parse_document_structure(
    pages: list[PageRecord],
) -> list[HeadingRecord]:

    headings = []
    current_document_id = None
    state = HeadingState()

    for page in pages:

        if page.document_id != current_document_id:
            current_document_id = page.document_id
            state = HeadingState()

        for line_spans in group_spans_by_line(page.spans):

            heading = classify_heading_line(line_spans)

            if heading is None:
                continue

            update_heading_state(
                state=state,
                heading=heading,
            )

            headings.append(
                HeadingRecord(
                    document_id=page.document_id,
                    page_number=page.page_number,
                    heading_type=heading["type"],
                    number=heading["number"],
                    title=heading["title"],
                    section_number=state.section_number,
                    section_title=state.section_title,
                    subsection_number=state.subsection_number,
                    subsection_title=state.subsection_title,
                )
            )

    return headings


def _span_center_inside_bbox(
    span: TextSpanRecord,
    bbox: tuple[float, float, float, float],
) -> bool:
    """Return whether a span's center lies inside a table region."""

    span_x0, span_y0, span_x1, span_y1 = span.bbox
    box_x0, box_y0, box_x1, box_y1 = bbox
    center_x = (span_x0 + span_x1) / 2
    center_y = (span_y0 + span_y1) / 2

    return (
        box_x0 <= center_x <= box_x1
        and box_y0 <= center_y <= box_y1
    )


def map_table_heading_states(
    pages: list[PageRecord],
    tables: list[TableRecord],
) -> dict[tuple[str, int, int], HeadingState]:
    """Capture the active heading state at every table position."""

    tables_by_page: dict[
        tuple[str, int],
        list[TableRecord],
    ] = {}

    seen_table_keys: set[tuple[str, int, int]] = set()

    for table in tables:
        table_key = (
            table.document_id,
            table.page_number,
            table.table_index,
        )

        if table_key in seen_table_keys:
            raise ValueError(
                "Duplicate table key: "
                f"{table_key}"
            )

        seen_table_keys.add(table_key)
        page_key = (
            table.document_id,
            table.page_number,
        )
        tables_by_page.setdefault(
            page_key,
            [],
        ).append(table)

    table_states: dict[
        tuple[str, int, int],
        HeadingState,
    ] = {}

    current_document_id = None
    state = HeadingState()

    for page in pages:
        if page.document_id != current_document_id:
            current_document_id = page.document_id
            state = HeadingState()

        page_key = (
            page.document_id,
            page.page_number,
        )
        page_tables = sorted(
            tables_by_page.get(page_key, []),
            key=lambda table: (
                table.bbox[1],
                table.table_index,
            ),
        )

        heading_events: list[
            tuple[float, dict[str, str]]
        ] = []

        for line_spans in group_spans_by_line(
            page.spans
        ):
            # A bold numeric table cell can resemble a policy heading. It must
            # not alter the section state used by this or later tables.
            if any(
                _span_center_inside_bbox(
                    span,
                    table.bbox,
                )
                for span in line_spans
                for table in page_tables
            ):
                continue

            heading = classify_heading_line(
                line_spans
            )

            if heading is None:
                continue

            heading_top = min(
                span.bbox[1]
                for span in line_spans
            )
            heading_events.append(
                (
                    heading_top,
                    heading,
                )
            )

        heading_events.sort(
            key=lambda event: event[0]
        )

        heading_position = 0

        for table in page_tables:
            while (
                heading_position
                < len(heading_events)
                and heading_events[heading_position][0]
                <= table.bbox[1]
            ):
                update_heading_state(
                    state=state,
                    heading=heading_events[
                        heading_position
                    ][1],
                )
                heading_position += 1

            table_key = (
                table.document_id,
                table.page_number,
                table.table_index,
            )
            table_states[table_key] = HeadingState(
                section_number=state.section_number,
                section_title=state.section_title,
                subsection_number=(
                    state.subsection_number
                ),
                subsection_title=(
                    state.subsection_title
                ),
            )

        # Headings below the final table still need to update
        # the state that continues onto the next page.
        while heading_position < len(
            heading_events
        ):
            update_heading_state(
                state=state,
                heading=heading_events[
                    heading_position
                ][1],
            )
            heading_position += 1

    return table_states

def attach_table_heading_states(
    pages: list[PageRecord],
    tables: list[TableRecord],
) -> None:
    """
    Persist the active section/subsection context directly
    onto every TableRecord.

    Reuses map_table_heading_states() so heading detection
    logic exists in only one place.
    """

    table_states = map_table_heading_states(
        pages=pages,
        tables=tables,
    )

    for table in tables:
        table_key = (
            table.document_id,
            table.page_number,
            table.table_index,
        )

        state = table_states.get(
            table_key
        )

        if state is None:
            raise ValueError(
                "Missing heading state for table: "
                f"{table_key}"
            )

        table.section_number = (
            state.section_number
        )

        table.section_title = (
            state.section_title
        )

        table.subsection_number = (
            state.subsection_number
        )

        table.subsection_title = (
            state.subsection_title
        )