import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict


def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - remove extra whitespace
    - remove line breaks
    """
    return " ".join(text.split())


def load_pdfs(folder_path: str) -> List[Dict]:
    """
    Load all PDFs and return page-level documents.

    Output format:
    [
        {
            "doc_id": "code_of_conduct",
            "page": 1,
            "text": "...."
        }
    ]
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    documents = []

    pdf_files = list(folder.glob("*.pdf"))

    if not pdf_files:
        raise ValueError("No PDF files found in folder")

    total_pages = 0

    for pdf_path in pdf_files:
        doc_id = pdf_path.stem.lower()

        print(f"[LOAD] {pdf_path.name}")

        pdf = fitz.open(pdf_path)

        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text = page.get_text()

            if not text or len(text.strip()) < 20:
                continue  # skip empty pages

            cleaned = clean_text(text)

            documents.append({
                "doc_id": doc_id,
                "page": page_num + 1,
                "text": cleaned
            })

            total_pages += 1

        pdf.close()

    print(f"[LOAD] {total_pages} pages loaded\n")

    return documents