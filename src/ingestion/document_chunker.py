from src.core.models import ChunkRecord, StructuredDocumentRecord
from src.ingestion.chunker import ChunkingConfig, chunk_body_content
from src.ingestion.table_chunker import chunk_document_tables


def _validate_document_provenance(
    document: StructuredDocumentRecord,
) -> None:
    """Reject inconsistent document metadata before creating chunks."""

    metadata = document.metadata

    for record in document.body_content:
        if (
            record.document_id != metadata.document_id
            or record.document_name != metadata.document_name
            or record.source_file != metadata.source_file
        ):
            raise ValueError(
                "Body record provenance does not match document metadata: "
                f"{metadata.document_id}, page {record.page_number}"
            )

    for table in document.tables:
        if table.document_id != metadata.document_id:
            raise ValueError(
                "Table provenance does not match document metadata: "
                f"{metadata.document_id}, page {table.page_number}"
            )


def chunk_document(
    document: StructuredDocumentRecord,
    config: ChunkingConfig | None = None,
) -> list[ChunkRecord]:
    """Create all retrieval chunks for one structured document."""

    active_config = config or ChunkingConfig()
    _validate_document_provenance(document)
    body_chunks = chunk_body_content(
        records=document.body_content,
        config=active_config,
        starting_index=1,
    )
    table_chunks = chunk_document_tables(
        document=document,
        config=active_config,
        starting_index=len(body_chunks) + 1,
    )

    # Retrieval does not depend on physical list order. Keeping body and table
    # chunks in separate deterministic groups avoids guessing whether a body
    # block occurred above or below a table when a body chunk spans the table.
    return [*body_chunks, *table_chunks]


def chunk_documents(
    documents: list[StructuredDocumentRecord],
    config: ChunkingConfig | None = None,
) -> list[ChunkRecord]:
    """Create one flat deterministic chunk list from ordered documents."""

    active_config = config or ChunkingConfig()
    chunks: list[ChunkRecord] = []
    seen_document_ids: set[str] = set()

    for document in documents:
        document_id = document.metadata.document_id

        if document_id in seen_document_ids:
            raise ValueError(
                "Duplicate document ID: "
                f"{document_id}"
            )

        seen_document_ids.add(document_id)
        chunks.extend(
            chunk_document(
                document=document,
                config=active_config,
            )
        )

    return chunks
