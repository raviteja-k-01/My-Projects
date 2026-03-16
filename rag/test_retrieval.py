from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)

print("Loading vector database...")

db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

print("Vector DB loaded successfully.\n")

while True:
    query = input("Ask a question: ")

    if query.lower() in ["exit", "quit"]:
        break

    results = db.similarity_search(query, k=5)

    print("\nTop retrieved chunks:\n")

    for i, doc in enumerate(results):
        print(f"\nResult {i+1}")
        print("-" * 40)
        print(doc.page_content[:500])