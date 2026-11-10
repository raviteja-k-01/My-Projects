from src.ingestion.pipeline import process_and_export


def main() -> None:
    """Create the final Step 7 files from all source PDFs."""

    result = process_and_export()

    print(f"Documents: {result.document_count}")
    print(f"Tables: {result.table_count}")
    print(f"Body chunks: {result.body_chunk_count}")
    print(f"Table chunks: {result.table_chunk_count}")
    print(f"Total chunks: {result.chunk_count}")
    print(f"Saved: {result.documents_path}")
    print(f"Saved: {result.tables_path}")
    print(f"Saved: {result.chunks_path}")
    print("Ingestion Step 7 export: PASS")


if __name__ == "__main__":
    main()
