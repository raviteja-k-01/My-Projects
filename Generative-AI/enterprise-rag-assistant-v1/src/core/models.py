from dataclasses import dataclass


@dataclass
class TextSpanRecord:
    text: str
    font_size: float
    font_name: str
    bbox: tuple[float, float, float, float]

    # PyMuPDF reading-order coordinates. Defaults keep older call sites that
    # construct spans directly working while newly extracted spans retain the
    # original block/line/span boundaries needed for reconstruction.
    block_index: int = 0
    line_index: int = 0
    span_index: int = 0


@dataclass
class PageRecord:
    document_id: str
    document_name: str
    page_number: int
    text: str
    source_file: str

    page_width: float
    page_height: float

    spans: list[TextSpanRecord]


@dataclass
class TableRecord:
    document_id: str
    page_number: int
    table_index: int

    caption: str | None

    rows: list[list[str]]
    bbox: tuple[float, float, float, float]

    searchable_text: str = ""

    section_number: str | None = None
    section_title: str | None = None
    subsection_number: str | None = None
    subsection_title: str | None = None
@dataclass
class HeadingRecord:
    document_id: str
    page_number: int
    heading_type: str

    number: str
    title: str

    section_number: str | None
    section_title: str | None

    subsection_number: str | None
    subsection_title: str | None


@dataclass
class BodyContentRecord:
    """One reconstructed, retrieval-ready body paragraph or bullet."""

    document_id: str
    document_name: str
    page_number: int
    content_type: str
    text: str

    section_number: str | None = None
    section_title: str | None = None
    subsection_number: str | None = None
    subsection_title: str | None = None

    source_file: str = ""
    block_index: int = 0


@dataclass
class DocumentMetadataRecord:
    """Canonical controlled metadata extracted from document front matter."""

    document_id: str
    document_name: str
    source_file: str

    version: str | None = None
    effective_date: str | None = None
    review_date: str | None = None
    classification: str | None = None
    owner: str | None = None
    approver: str | None = None
    department: str | None = None
    applicability: str | None = None
    status: str | None = None

    @property
    def title(self) -> str:
        return self.document_name


@dataclass
class StructuredDocumentRecord:
    """Steps 1-5 output; intentionally contains no chunks or embeddings."""

    metadata: DocumentMetadataRecord
    pages: list[PageRecord]
    headings: list[HeadingRecord]
    body_content: list[BodyContentRecord]
    tables: list[TableRecord]

@dataclass
class ChunkRecord:
    """One section-aware retrieval unit created by Step 6."""

    chunk_id: str
    chunk_index: int

    document_id: str
    document_name: str
    source_file: str

    content_type: str
    text: str
    token_count: int

    page_start: int
    page_end: int

    section_number: str | None = None
    section_title: str | None = None
    subsection_number: str | None = None
    subsection_title: str | None = None

    table_index: int | None = None
    table_caption: str | None = None
    table_row_start: int | None = None
    table_row_end: int | None = None
