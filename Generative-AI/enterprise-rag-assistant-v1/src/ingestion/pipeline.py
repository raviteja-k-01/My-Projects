from __future__ import annotations

from pathlib import Path

import pymupdf

from src.core.models import ChunkRecord, StructuredDocumentRecord, TableRecord
from src.ingestion.body_reconstructor import reconstruct_body_content
from src.ingestion.chunker import ChunkingConfig
from src.ingestion.document_chunker import chunk_documents
from src.ingestion.exporter import (
    PROCESSED_DATA_DIR,
    ExportResult,
    export_ingestion_corpus,
)
from src.ingestion.load_documents import RAW_DATA_DIR, get_document_info
from src.ingestion.metadata_extractor import extract_document_metadata
from src.ingestion.pdf_reader import read_pdf
from src.ingestion.structure_parser import (
    attach_table_heading_states,
    parse_document_structure,
)
from src.ingestion.table_extractor import extract_tables


def _extract_document_tables(
    pdf_path: Path,
    document_id: str,
) -> list[TableRecord]:
    tables: list[TableRecord] = []

    with pymupdf.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            tables.extend(
                extract_tables(
                    page=page,
                    document_id=document_id,
                    page_number=page_index + 1,
                )
            )

    return tables


def process_document(pdf_path: Path) -> StructuredDocumentRecord:
    """Run the existing extraction flow through ingestion steps 1-5."""

    document_id, fallback_document_name = get_document_info(pdf_path)
    pages = read_pdf(
        pdf_path=pdf_path,
        document_id=document_id,
        document_name=fallback_document_name,
    )
    tables = _extract_document_tables(
        pdf_path=pdf_path,
        document_id=document_id,
    )
    metadata = extract_document_metadata(
        tables=tables,
        fallback_document_id=document_id,
        fallback_document_name=fallback_document_name,
        source_file=pdf_path.name,
    )

    # Replace the filename-derived display name with the controlled title before
    # reconstructed body records inherit document metadata.
    for page in pages:
        page.document_name = metadata.document_name

        headings = parse_document_structure(pages)

        attach_table_heading_states(
            pages=pages,
            tables=tables,
        )

        body_content = reconstruct_body_content(
            pages=pages,
            tables=tables,
        )

    return StructuredDocumentRecord(
        metadata=metadata,
        pages=pages,
        headings=headings,
        body_content=body_content,
        tables=tables,
    )


def process_documents(
    raw_data_dir: Path = RAW_DATA_DIR,
) -> list[StructuredDocumentRecord]:
    """Process every source PDF through Step 5, in filename order."""

    return [
        process_document(pdf_path)
        for pdf_path in sorted(raw_data_dir.glob("*.pdf"))
    ]


def process_chunks(
    raw_data_dir: Path = RAW_DATA_DIR,
    config: ChunkingConfig | None = None,
) -> list[ChunkRecord]:
    """Run ingestion through Step 6 and return one flat chunk list."""

    return chunk_documents(
        documents=process_documents(raw_data_dir),
        config=config,
    )


def process_and_export(
    raw_data_dir: Path = RAW_DATA_DIR,
    output_dir: Path = PROCESSED_DATA_DIR,
    config: ChunkingConfig | None = None,
) -> ExportResult:
    """Run ingestion through Step 7 and save the final JSONL corpus."""

    documents = process_documents(raw_data_dir)
    return export_ingestion_corpus(
        documents=documents,
        output_dir=output_dir,
        config=config,
    )
