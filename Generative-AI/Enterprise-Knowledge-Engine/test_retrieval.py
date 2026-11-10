from app.core.hybrid_retriever import HybridRetriever

retriever = HybridRetriever()

query = "What is Amazon S3?"

results = retriever.search(query)

for r in results:
    print("\n---")
    print("Score:", r["score"])
    print("Source:", r["source"])
    print("Text:", r["text"][:200])