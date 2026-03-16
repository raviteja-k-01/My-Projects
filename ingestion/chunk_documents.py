import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH = "data/pdfs"


def load_documents():

    documents = []

    files = os.listdir(DATA_PATH)
    pdf_files = [f for f in files if f.endswith(".pdf")]

    print("Files found:", pdf_files)

    for file in pdf_files:

        path = os.path.join(DATA_PATH, file)

        loader = PyPDFLoader(path)
        pages = loader.load()

        print(f"Loaded {file} with {len(pages)} pages")

        documents.extend(pages)

    print("\nTotal pages loaded:", len(documents))

    return documents


def chunk_documents():

    documents = load_documents()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\nCHAPTER",
            "\nChapter",
            "\nPart",
            "\nSection",
            "\n\n",
            "\n",
            ". ",
            " "
        ]
    )

    chunks = text_splitter.split_documents(documents)

    print("\nTotal chunks created:", len(chunks))

    print("\nSample chunk:\n")
    print(chunks[0].page_content[:500])

    print("\nMetadata:")
    print(chunks[0].metadata)

    return chunks


if __name__ == "__main__":
    chunk_documents()