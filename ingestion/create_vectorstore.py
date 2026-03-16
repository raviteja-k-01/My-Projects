from chunk_documents import chunk_documents
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# load already cleaned + chunked documents
chunks = chunk_documents()

print("Total chunks:", len(chunks))


print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)


print("Creating FAISS vector database...")

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

vectorstore.save_local("vectorstore")

print("Vector database created successfully.")