import streamlit as st
import ollama
import time
import requests
import subprocess

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi


# -----------------------------
# CONFIG
# -----------------------------

VECTOR_DB_PATH = "vectorstore"

EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-large"
LLM_MODEL = "qwen2.5:14b"

OLLAMA_PATH = r"C:\Users\PMF_AI_R&D\AppData\Local\Programs\Ollama\ollama.exe"

RETRIEVAL_K = 60
FINAL_DOCS = 5


# -----------------------------
# START OLLAMA
# -----------------------------

def ensure_ollama():

    try:
        requests.get("http://localhost:11434")
    except:
        subprocess.Popen(
            [OLLAMA_PATH, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(4)


# -----------------------------
# LOAD SYSTEM ONCE
# -----------------------------

@st.cache_resource
def load_system():

    ensure_ollama()

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    db = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    reranker = CrossEncoder(RERANKER_MODEL)

    docs = db.similarity_search("", k=2000)

    corpus = [d.page_content for d in docs]
    tokenized = [doc.split(" ") for doc in corpus]

    bm25 = BM25Okapi(tokenized)

    return db, reranker, bm25, docs


# -----------------------------
# RETRIEVE
# -----------------------------

def retrieve(question, db, reranker, bm25, bm25_docs):

    vector_docs = db.similarity_search(question, k=40)

    tokenized_query = question.split(" ")

    scores = bm25.get_scores(tokenized_query)

    top_idx = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:20]

    bm25_results = [bm25_docs[i] for i in top_idx]

    combined = vector_docs + bm25_results

    unique_docs = list({d.page_content: d for d in combined}.values())

    pairs = [(question, doc.page_content) for doc in unique_docs]

    rerank_scores = reranker.predict(pairs)

    ranked = sorted(zip(rerank_scores, unique_docs), key=lambda x: x[0], reverse=True)

    return [doc for _, doc in ranked[:FINAL_DOCS]]


# -----------------------------
# GENERATE ANSWER
# -----------------------------

def generate_answer(question, context):

    prompt = f"""
You are an expert assistant on RBI financial regulations.

Use ONLY the provided context.

Give:
1. Short definition
2. Key regulatory points

Context:
{context}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


# -----------------------------
# UI
# -----------------------------

st.set_page_config(
    page_title="RBI Financial AI",
    page_icon="📊",
    layout="wide"
)

st.title("📊 RBI Financial Regulations AI Assistant")

st.write(
    "Ask questions about RBI regulations, Basel III, KYC, Priority Sector Lending and other financial rules."
)

# Load system
db, reranker, bm25, bm25_docs = load_system()

question = st.text_input("Ask your question")

if st.button("Ask"):

    if question:

        with st.spinner("Searching regulations and generating answer..."):

            docs = retrieve(question, db, reranker, bm25, bm25_docs)

            context = "\n\n".join([doc.page_content for doc in docs])

            sources = [
                f"{doc.metadata.get('source')} (page {doc.metadata.get('page')})"
                for doc in docs
            ]

            answer = generate_answer(question, context)

        st.success("Answer generated")

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")

        for s in sources:
            st.write("•", s)