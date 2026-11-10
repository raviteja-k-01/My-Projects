from pathlib import Path

from src.core.models import PageRecord
from src.ingestion.pdf_reader import read_pdf


RAW_DATA_DIR = Path("data/raw")


def get_document_info(pdf_path: Path) -> tuple[str, str]:
    """
    Example filename:
    02_HR-LV-07_Leave_Attendance.pdf
    """

    parts = pdf_path.stem.split("_", 2)

    document_id = parts[1]
    document_name = parts[2].replace("_", " ")

    return document_id, document_name


def load_documents() -> list[PageRecord]:

    all_pages = []

    pdf_files = sorted(RAW_DATA_DIR.glob("*.pdf"))

    for pdf_path in pdf_files:

        document_id, document_name = get_document_info(pdf_path)

        pages = read_pdf(
            pdf_path=pdf_path,
            document_id=document_id,
            document_name=document_name,
        )

        all_pages.extend(pages)

    return all_pages


if __name__ == "__main__":

    pages = load_documents()

    document_ids = {page.document_id for page in pages}

    print(f"Total documents: {len(document_ids)}")
    print(f"Total extracted pages: {len(pages)}")