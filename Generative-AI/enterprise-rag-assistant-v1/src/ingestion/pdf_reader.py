from pathlib import Path

import pymupdf

from src.core.models import PageRecord, TextSpanRecord


def extract_spans(page: pymupdf.Page) -> list[TextSpanRecord]:

    spans = []

    page_dict = page.get_text("dict")

    for block_index, block in enumerate(page_dict["blocks"]):

        if "lines" not in block:
            continue

        for line_index, line in enumerate(block["lines"]):

            for span_index, span in enumerate(line["spans"]):

                text = span["text"].strip()

                if not text:
                    continue

                spans.append(
                    TextSpanRecord(
                        text=text,
                        font_size=span["size"],
                        font_name=span["font"],
                        bbox=tuple(span["bbox"]),
                        block_index=block_index,
                        line_index=line_index,
                        span_index=span_index,
                    )
                )

    return spans


def read_pdf(
    pdf_path: Path,
    document_id: str,
    document_name: str,
) -> list[PageRecord]:

    pages = []

    document = pymupdf.open(pdf_path)

    for page_index, page in enumerate(document):

        page_record = PageRecord(
            document_id=document_id,
            document_name=document_name,
            page_number=page_index + 1,
            text=page.get_text(),
            source_file=pdf_path.name,
            page_width=page.rect.width,
            page_height=page.rect.height,
            spans=extract_spans(page),
        )

        pages.append(page_record)

    document.close()

    return pages
