import re

import pymupdf

from src.core.models import TableRecord
from src.ingestion.table_search import table_to_searchable_text
from src.ingestion.text_normalizer import join_wrapped_lines


def clean_cell(cell: str | None) -> str:
    if cell is None:
        return ""

    return join_wrapped_lines(cell.splitlines())


def is_spillover_row(row: list[str]) -> bool:
    """
    Detect narrative text that PyMuPDF accidentally
    attaches as a merged table row.
    """

    if len(row) <= 1:
        return False

    first_cell = row[0].strip() if row[0] else ""

    remaining_cells = [
        cell.strip() if cell else ""
        for cell in row[1:]
    ]

    return (
        len(first_cell) >= 120
        and all(not cell for cell in remaining_cells)
    )


def get_effective_table_bbox(
    table,
    has_spillover: bool,
) -> tuple[float, float, float, float]:

    raw_bbox = tuple(table.bbox)

    if not has_spillover:
        return raw_bbox

    x0, y0, x1, _ = raw_bbox

    table_width = x1 - x0

    normal_cells = []

    for cell in table.cells:
        if cell is None:
            continue

        cx0, cy0, cx1, cy1 = cell

        cell_width = cx1 - cx0

        if cell_width >= table_width * 0.90:
            continue

        normal_cells.append(cell)

    if not normal_cells:
        return raw_bbox

    effective_bottom = max(
        cell[3]
        for cell in normal_cells
    )

    return (
        x0,
        y0,
        x1,
        effective_bottom,
    )


def find_table_caption(
    page: pymupdf.Page,
    table_bbox: tuple[float, float, float, float],
    document_id: str,
) -> str | None:

    table_top = table_bbox[1]

    page_dict = page.get_text("dict")

    candidates = []

    pattern = re.compile(
        rf"^{re.escape(document_id)}-[A-Z]{{1,3}}\d{{0,2}}\b"
    )

    for block in page_dict["blocks"]:
        if "lines" not in block:
            continue

        for line in block["lines"]:
            for span in line["spans"]:

                text = span["text"].strip()

                if not text:
                    continue

                _, _, _, y1 = span["bbox"]

                distance = table_top - y1

                if not (0 <= distance <= 30):
                    continue

                if "Bold" not in span["font"]:
                    continue

                if not pattern.match(text):
                    continue

                candidates.append(
                    (
                        distance,
                        text,
                    )
                )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0]
    )

    return candidates[0][1]


def extract_tables(
    page: pymupdf.Page,
    document_id: str,
    page_number: int,
) -> list[TableRecord]:

    table_records = []

    detected_tables = page.find_tables()

    for table_index, table in enumerate(
        detected_tables.tables,
        start=1,
    ):

        rows = table.extract()

        cleaned_rows = []

        has_spillover = False

        for row in rows:

            cleaned_row = [
                clean_cell(cell)
                for cell in row
            ]

            if is_spillover_row(cleaned_row):
                has_spillover = True
                continue

            cleaned_rows.append(cleaned_row)

        effective_bbox = get_effective_table_bbox(
            table=table,
            has_spillover=has_spillover,
        )

        caption = find_table_caption(
            page=page,
            table_bbox=tuple(table.bbox),
            document_id=document_id,
        )

        table_record = TableRecord(
            document_id=document_id,
            page_number=page_number,
            table_index=table_index,
            caption=caption,
            rows=cleaned_rows,
            bbox=effective_bbox,
        )

        table_record.searchable_text = table_to_searchable_text(table_record)

        table_records.append(table_record)

    return table_records
