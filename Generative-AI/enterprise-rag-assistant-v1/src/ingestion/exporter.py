from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from src.core.models import (
    ChunkRecord,
    DocumentMetadataRecord,
    StructuredDocumentRecord,
    TableRecord,
)
from src.ingestion.chunker import ChunkingConfig, estimate_token_count
from src.ingestion.document_chunker import chunk_documents
from src.ingestion.table_search import table_to_searchable_text


PROCESSED_DATA_DIR = Path("data/processed")
DOCUMENTS_FILENAME = "documents.jsonl"
TABLES_FILENAME = "tables.jsonl"
CHUNKS_FILENAME = "chunks.jsonl"


@dataclass(frozen=True)
class ExportResult:
    """Paths and counts produced by one completed Step 7 export."""

    documents_path: Path
    tables_path: Path
    chunks_path: Path
    document_count: int
    table_count: int
    chunk_count: int
    body_chunk_count: int
    table_chunk_count: int


def _validate_metadata(metadata: DocumentMetadataRecord) -> None:
    """Validate fields needed to identify and join exported records."""

    if not metadata.document_id.strip():
        raise ValueError("Document ID cannot be empty")

    if not metadata.document_name.strip():
        raise ValueError(
            f"Document name cannot be empty: {metadata.document_id}"
        )

    if not metadata.source_file.strip():
        raise ValueError(
            f"Source file cannot be empty: {metadata.document_id}"
        )

    for field_name in ("effective_date", "review_date"):
        field_value = getattr(metadata, field_name)

        if field_value is not None:
            parsed_date = date.fromisoformat(field_value)

            if parsed_date.isoformat() != field_value:
                raise ValueError(
                    f"{field_name} must use YYYY-MM-DD: "
                    f"{metadata.document_id}"
                )


def _validate_documents(
    documents: list[StructuredDocumentRecord],
) -> list[StructuredDocumentRecord]:
    """Return canonical source order after validating Step 5 records."""

    if not documents:
        raise ValueError("At least one document is required for export")

    ordered_documents = sorted(
        documents,
        key=lambda document: (
            document.metadata.source_file.casefold(),
            document.metadata.document_id,
        ),
    )
    document_ids = [
        document.metadata.document_id
        for document in ordered_documents
    ]

    if len(document_ids) != len(set(document_ids)):
        raise ValueError("Duplicate document IDs cannot be exported")

    seen_table_keys: set[tuple[str, int, int]] = set()

    for document in ordered_documents:
        metadata = document.metadata
        _validate_metadata(metadata)

        page_numbers = [
            page.page_number
            for page in document.pages
        ]
        expected_page_numbers = list(
            range(1, len(document.pages) + 1)
        )

        if page_numbers != expected_page_numbers:
            raise ValueError(
                "Document pages must be complete and ordered: "
                f"{metadata.document_id}"
            )

        for page in document.pages:
            if (
                page.document_id != metadata.document_id
                or page.document_name != metadata.document_name
                or page.source_file != metadata.source_file
            ):
                raise ValueError(
                    "Page provenance does not match document metadata: "
                    f"{metadata.document_id}, page {page.page_number}"
                )

        for table in document.tables:
            table_key = (
                table.document_id,
                table.page_number,
                table.table_index,
            )

            if table_key in seen_table_keys:
                raise ValueError(
                    f"Duplicate table key cannot be exported: {table_key}"
                )

            seen_table_keys.add(table_key)

            if table.document_id != metadata.document_id:
                raise ValueError(
                    "Table provenance does not match document metadata: "
                    f"{table_key}"
                )

            if table.page_number not in page_numbers:
                raise ValueError(
                    f"Table page is outside its document: {table_key}"
                )

            if len(table.bbox) != 4:
                raise ValueError(
                    f"Table bbox must contain four values: {table_key}"
                )

            if not all(math.isfinite(value) for value in table.bbox):
                raise ValueError(
                    f"Table bbox contains a non-finite value: {table_key}"
                )

            if table.searchable_text != table_to_searchable_text(table):
                raise ValueError(
                    f"Table searchable text is stale or invalid: {table_key}"
                )

    return ordered_documents


def _validate_chunks(
    documents: list[StructuredDocumentRecord],
    chunks: list[ChunkRecord],
    config: ChunkingConfig,
) -> None:
    """Validate the final Step 6 records before any file is replaced."""

    documents_by_id = {
        document.metadata.document_id: document
        for document in documents
    }
    table_keys = {
        (
            table.document_id,
            table.page_number,
            table.table_index,
        )
        for document in documents
        for table in document.tables
    }
    chunk_ids = [chunk.chunk_id for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs cannot be exported")

    chunks_by_document: dict[str, list[ChunkRecord]] = defaultdict(list)
    chunked_table_keys: set[tuple[str, int, int]] = set()

    for chunk in chunks:
        if chunk.document_id not in documents_by_id:
            raise ValueError(
                "Chunk refers to an unknown document: "
                f"{chunk.chunk_id}"
            )

        document = documents_by_id[chunk.document_id]
        metadata = document.metadata
        valid_pages = {
            page.page_number
            for page in document.pages
        }

        if (
            chunk.document_name != metadata.document_name
            or chunk.source_file != metadata.source_file
        ):
            raise ValueError(
                "Chunk provenance does not match document metadata: "
                f"{chunk.chunk_id}"
            )

        if (
            chunk.page_start not in valid_pages
            or chunk.page_end not in valid_pages
            or chunk.page_start > chunk.page_end
        ):
            raise ValueError(
                f"Chunk page range is invalid: {chunk.chunk_id}"
            )

        if not chunk.text.strip():
            raise ValueError(
                f"Chunk text cannot be empty: {chunk.chunk_id}"
            )

        if chunk.token_count != estimate_token_count(chunk.text):
            raise ValueError(
                f"Chunk token count is stale: {chunk.chunk_id}"
            )

        if chunk.token_count > config.max_tokens:
            raise ValueError(
                f"Chunk exceeds max_tokens: {chunk.chunk_id}"
            )

        if chunk.content_type == "body":
            if any(
                value is not None
                for value in (
                    chunk.table_index,
                    chunk.table_caption,
                    chunk.table_row_start,
                    chunk.table_row_end,
                )
            ):
                raise ValueError(
                    f"Body chunk contains table metadata: {chunk.chunk_id}"
                )
        elif chunk.content_type == "table":
            if (
                chunk.page_start != chunk.page_end
                or chunk.table_index is None
            ):
                raise ValueError(
                    f"Table chunk metadata is invalid: {chunk.chunk_id}"
                )

            table_key = (
                chunk.document_id,
                chunk.page_start,
                chunk.table_index,
            )

            if table_key not in table_keys:
                raise ValueError(
                    "Table chunk refers to an unknown table: "
                    f"{chunk.chunk_id}"
                )

            chunked_table_keys.add(table_key)
        else:
            raise ValueError(
                f"Unsupported chunk content type: {chunk.chunk_id}"
            )

        chunks_by_document[chunk.document_id].append(chunk)

    if chunked_table_keys != table_keys:
        raise ValueError("Not every table produced a table chunk")

    for document in documents:
        document_id = document.metadata.document_id
        document_chunks = chunks_by_document.get(document_id, [])

        if not document_chunks:
            raise ValueError(
                f"Document produced no chunks: {document_id}"
            )

        expected_indexes = list(
            range(1, len(document_chunks) + 1)
        )
        actual_indexes = [
            chunk.chunk_index
            for chunk in document_chunks
        ]

        if actual_indexes != expected_indexes:
            raise ValueError(
                "Chunk indexes are not contiguous: "
                f"{document_id}"
            )


def _document_export_record(
    metadata: DocumentMetadataRecord,
) -> dict[str, Any]:
    """Flatten canonical document metadata and include its title alias."""

    record = asdict(metadata)
    record["title"] = metadata.title
    return record


def _serialize_jsonl(records: list[dict[str, Any]]) -> str:
    """Serialize records as deterministic UTF-8-compatible JSON lines."""

    return "".join(
        json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for record in records
    )


def _prepare_temp_file(
    destination: Path,
    contents: str,
) -> Path:
    """Write and flush a same-directory temporary export file."""

    temporary_path: Path | None = None

    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(contents)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            return temporary_path
    except BaseException:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise


def _replace_export_files(
    files: list[tuple[Path, str]],
) -> None:
    """Prepare every file first, then atomically replace each final file."""

    prepared_files: list[tuple[Path, Path]] = []

    try:
        for destination, contents in files:
            temporary_path = _prepare_temp_file(
                destination=destination,
                contents=contents,
            )
            prepared_files.append(
                (destination, temporary_path)
            )

        for destination, temporary_path in prepared_files:
            os.replace(temporary_path, destination)
    finally:
        for _, temporary_path in prepared_files:
            if temporary_path.exists():
                temporary_path.unlink()


def export_ingestion_corpus(
    documents: list[StructuredDocumentRecord],
    output_dir: Path = PROCESSED_DATA_DIR,
    config: ChunkingConfig | None = None,
) -> ExportResult:
    """Validate and save the completed ingestion corpus as three JSONL files."""

    active_config = config or ChunkingConfig()
    ordered_documents = _validate_documents(documents)
    chunks = chunk_documents(
        documents=ordered_documents,
        config=active_config,
    )
    _validate_chunks(
        documents=ordered_documents,
        chunks=chunks,
        config=active_config,
    )

    tables = [
        table
        for document in ordered_documents
        for table in sorted(
            document.tables,
            key=lambda table: (
                table.page_number,
                table.bbox[1],
                table.table_index,
            ),
        )
    ]
    document_records = [
        _document_export_record(document.metadata)
        for document in ordered_documents
    ]
    table_records = [asdict(table) for table in tables]
    chunk_records = [asdict(chunk) for chunk in chunks]

    # Complete serialization happens before any destination file is touched.
    documents_jsonl = _serialize_jsonl(document_records)
    tables_jsonl = _serialize_jsonl(table_records)
    chunks_jsonl = _serialize_jsonl(chunk_records)

    output_dir.mkdir(parents=True, exist_ok=True)
    documents_path = output_dir / DOCUMENTS_FILENAME
    tables_path = output_dir / TABLES_FILENAME
    chunks_path = output_dir / CHUNKS_FILENAME

    _replace_export_files(
        [
            (documents_path, documents_jsonl),
            (tables_path, tables_jsonl),
            (chunks_path, chunks_jsonl),
        ]
    )

    body_chunk_count = sum(
        chunk.content_type == "body"
        for chunk in chunks
    )
    table_chunk_count = sum(
        chunk.content_type == "table"
        for chunk in chunks
    )

    return ExportResult(
        documents_path=documents_path,
        tables_path=tables_path,
        chunks_path=chunks_path,
        document_count=len(ordered_documents),
        table_count=len(tables),
        chunk_count=len(chunks),
        body_chunk_count=body_chunk_count,
        table_chunk_count=table_chunk_count,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load strict one-object-per-line JSONL for typed reload helpers."""

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                raise ValueError(
                    f"Blank JSONL line: {path}, line {line_number}"
                )

            record = json.loads(line)

            if not isinstance(record, dict):
                raise ValueError(
                    f"JSONL record must be an object: {path}, line {line_number}"
                )

            records.append(record)

    return records


def load_document_metadata(
    path: Path,
) -> list[DocumentMetadataRecord]:
    """Reload canonical document metadata from documents.jsonl."""

    metadata_records: list[DocumentMetadataRecord] = []

    for record in _read_jsonl(path):
        title = record.pop("title", None)
        metadata = DocumentMetadataRecord(**record)

        if title != metadata.title:
            raise ValueError(
                "Document title alias does not match document_name: "
                f"{metadata.document_id}"
            )

        metadata_records.append(metadata)

    return metadata_records


def load_tables(path: Path) -> list[TableRecord]:
    """Reload cleaned tables while restoring tuple bboxes."""

    tables: list[TableRecord] = []

    for record in _read_jsonl(path):
        record["bbox"] = tuple(record["bbox"])
        tables.append(TableRecord(**record))

    return tables


def load_chunks(path: Path) -> list[ChunkRecord]:
    """Reload final retrieval chunks from chunks.jsonl."""

    return [
        ChunkRecord(**record)
        for record in _read_jsonl(path)
    ]
