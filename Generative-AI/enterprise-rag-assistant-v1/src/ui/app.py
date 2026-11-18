"""Streamlit frontend. All Gemini and Qdrant credentials stay in FastAPI."""
from __future__ import annotations

import os

from dotenv import load_dotenv
import requests
import streamlit as st


load_dotenv()
API_BASE_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="📄",
    layout="centered",
)
st.title("Enterprise RAG Assistant")
st.caption("Ask questions about the project's fictional enterprise policy documents.")

with st.form("rag_query", clear_on_submit=False):
    question = st.text_area(
        "Your question",
        placeholder="Example: How should a SEV-1 security incident be handled?",
        max_chars=1000,
    )
    submitted = st.form_submit_button(
        "Ask", type="primary", use_container_width=True
    )

if submitted:
    clean_question = question.strip()
    if not clean_question:
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Searching policies and preparing an evidence-grounded answer..."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/ask",
                json={"question": clean_question},
                timeout=(10, 120),
            )
        except requests.RequestException:
            st.error("The RAG backend is unreachable. Try again later.")
            st.stop()

    if response.status_code != 200:
        try:
            payload = response.json()
            detail = payload.get("detail", "Request failed.")
        except (ValueError, AttributeError):
            detail = "The backend returned an unexpected error."
        if response.status_code == 429:
            st.warning("This demo is receiving too many requests. Try again later.")
        elif response.status_code == 503:
            st.error("The AI service is temporarily unavailable. " + str(detail))
        else:
            st.error(f"Request failed ({response.status_code}): {detail}")
        st.stop()

    try:
        result = response.json()
        answer = result["answer"]
    except (ValueError, KeyError, TypeError):
        st.error("The backend returned an invalid answer.")
        st.stop()

    retrieval_mode = result.get("retrieval_mode", "hybrid")
    if retrieval_mode == "bm25_fallback":
        st.warning(
            "Semantic search is temporarily unavailable. "
            "This answer uses keyword-only retrieval and may be less complete."
        )

    st.subheader("Answer")
    st.markdown(answer)

    citations = result.get("citations", [])
    if citations:
        st.subheader("Sources")
        for citation in citations:
            doc = citation.get("document_id", "Unknown document")
            page = citation.get("page_label", "Unknown page")
            section = citation.get("section_label", "Not specified")
            chunk = citation.get("chunk_id", "")
            st.markdown(
                f"**{doc}**  \nPage: {page}  \n"
                f"Section: {section}  \nChunk: `{chunk}`"
            )
    else:
        st.info("No source citations were returned. Verify this answer before relying on it.")

    latency = result.get("latency", {})
    with st.expander("Retrieval details"):
        st.write("Mode:", retrieval_mode)
        st.write("Evidence chunks:", result.get("evidence_count", "Unknown"))
        st.write("Generation model:", result.get("model", "Unknown"))
        seconds = latency.get("total_seconds")
        if isinstance(seconds, (int, float)):
            st.write(f"Total latency: {seconds:.2f} s")
