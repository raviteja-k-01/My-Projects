import os
import fitz
from langchain.schema import Document

DATA_PATH = "data/pdfs"

GARBAGE_PATTERNS = [
    "SELECT LANGUAGE",
    "SITE MAP",
    "CONTACT US",
    "Archives",
    "Print",
    "rbi.org.in",
    "Search",
    "Go",
]


def clean_text(text):

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if any(pattern.lower() in line.lower() for pattern in GARBAGE_PATTERNS):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def load_documents():

    documents = []

    files = os.listdir(DATA_PATH)
    pdf_files = [f for f in files if f.endswith(".pdf")]

    print("Files found:", pdf_files)

    for file in pdf_files:

        path = os.path.join(DATA_PATH, file)

        pdf = fitz.open(path)

        print(f"Loaded {file} with {len(pdf)} pages")

        for page_number, page in enumerate(pdf):

            text = page.get_text()

            text = clean_text(text)

            doc = Document(
                page_content=text,
                metadata={
                    "source": file,
                    "page": page_number + 1
                }
            )

            documents.append(doc)

    print("\nTotal pages loaded:", len(documents))

    return documents