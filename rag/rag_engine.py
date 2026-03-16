import os
import time
import subprocess
import requests
import ollama

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
# --------------------------------------------
# Configuration
# --------------------------------------------

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

OLLAMA_PATH = r"C:\Users\PMF_AI_R&D\AppData\Local\Programs\Ollama\ollama.exe"
VECTOR_DB_PATH = "vectorstore"

EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-large"
LLM_MODEL = "qwen2.5:14b"

RETRIEVAL_K = 60
FINAL_CONTEXT_DOCS = 5

# --------------------------------------------
# Ensure Ollama server is running
# --------------------------------------------

def ensure_ollama_running():
    try:
        requests.get("http://localhost:11434")
        return
    except:
        print("Starting Ollama server...")

        subprocess.Popen(
            [OLLAMA_PATH, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW
        )

        time.sleep(5)

# --------------------------------------------
# Load Models
# --------------------------------------------

def load_models():

    print("Loading reranker...")
    reranker = CrossEncoder(RERANKER_MODEL)

    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print("Loading vector database...")

    db = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("Building BM25 index...")

    docs = db.similarity_search("", k=2000)

    corpus = [doc.page_content for doc in docs]
    tokenized_corpus = [doc.split(" ") for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus)

    return db, reranker, bm25, docs

# --------------------------------------------
# Retrieve + Rerank Documents
# --------------------------------------------

def retrieve_documents(question, db, reranker, bm25, bm25_docs):

    # ------------------
    # Vector search
    # ------------------

    vector_docs = db.similarity_search(question, k=40)

    # ------------------
    # BM25 search
    # ------------------

    tokenized_query = question.split(" ")

    bm25_scores = bm25.get_scores(tokenized_query)

    top_bm25_idx = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:20]

    bm25_docs = [bm25_docs[i] for i in top_bm25_idx]

    # ------------------
    # Merge results
    # ------------------

    combined_docs = vector_docs + bm25_docs

    # remove duplicates
    unique_docs = list({doc.page_content: doc for doc in combined_docs}.values())

    # ------------------
    # Reranking
    # ------------------

    pairs = [(question, doc.page_content) for doc in unique_docs]

    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, unique_docs), key=lambda x: x[0], reverse=True)

    final_docs = [doc for _, doc in ranked[:FINAL_CONTEXT_DOCS]]

    return final_docs

# --------------------------------------------
# Build Context
# --------------------------------------------

def build_context(docs):

    context = "\n\n".join([doc.page_content for doc in docs])

    sources = []

    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        sources.append(f"{source} (page {page})")

    return context, sources

# --------------------------------------------
# Ask LLM
# --------------------------------------------

def generate_answer(question, context):

    prompt = f"""
You are an expert assistant on RBI financial regulations.

Use ONLY the information from the provided context.

Instructions:
• Give a short definition first.
• Then list key regulatory points.
• Quote relevant sentences when useful.
• Do not add information not present in the context.

If the answer truly does not exist in the context say:
"I cannot find this information in the provided documents."

Context:
{context}

Question:
{question}

Answer:
    """

    response = ollama.chat (
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
        options={"num_predict": 300}
    )

    return response["message"]["content"]

# --------------------------------------------
# Main RAG Loop
# --------------------------------------------

def main():

    ensure_ollama_running()

    db, reranker, bm25, bm25_docs = load_models()

    print("\nFinancial Regulations RAG System Ready.\n")

    while True:

        question = input("Ask a question: ")

        if question.lower() in ["exit", "quit"]:
            print("Exiting system.")
            break

        docs = retrieve_documents(question, db, reranker, bm25, bm25_docs)

        context, sources = build_context(docs)

        answer = generate_answer(question, context)

        print("\nAnswer:\n")
        print(answer)

        print("\nSources:")

        for s in sources:
            print("-", s)

        print("\n----------------------------\n")

# --------------------------------------------

if __name__ == "__main__":
    main()