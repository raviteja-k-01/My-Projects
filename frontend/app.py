import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/query"

st.set_page_config(page_title="Enterprise Knowledge Engine", layout="wide")

st.title("Enterprise Knowledge Engine")
st.subheader("Hybrid RAG System (BM25 + FAISS + Reranker + LLM)")

query = st.text_input("Enter your query:")

if st.button("Search"):
    if not query:
        st.warning("Please enter a query")
    else:
        with st.spinner("Processing..."):
            response = requests.get(API_URL, params={"q": query})
            data = response.json()

        st.markdown("## 🧠 Answer")
        st.write(data["answer"])

        st.markdown("## 📄 Sources")
        for i, src in enumerate(data["sources"]):
            with st.expander(f"Source {i+1} | Score: {round(src['rerank_score'], 2)}"):
                st.write(f"**Document:** {src['source']}")
                st.write(src["text"])

        st.markdown("## ⏱ Latency")
        st.write(f"{round(data['latency'], 2)} seconds")