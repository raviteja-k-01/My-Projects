import os

import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

API_BASE_URL = os.getenv(
    "RAG_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")


st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="📄",
    layout="centered",
)


st.title("Enterprise RAG Assistant")

st.caption(
    "Ask questions about enterprise policy documents."
)


question = st.text_area(
    "Your question",
    placeholder=(
        "Example: How should a SEV-1 "
        "security incident be handled?"
    ),
)


if st.button(
    "Ask",
    type="primary",
    use_container_width=True,
):

    if not question.strip():
        st.warning(
            "Please enter a question."
        )

    else:
        with st.spinner(
            "Searching documents and generating answer..."
        ):

            try:
                response = requests.post(
                    f"{API_BASE_URL}/ask",
                    json={
                        "question": question
                    },
                    timeout=120,
                )

            except requests.RequestException as exc:
                    st.error(
                        f"RAG API request failed: "
                        f"{type(exc).__name__}: {exc}"
                   )

            else:

                if response.status_code == 200:

                    result = response.json()

                    st.subheader("Answer")

                    st.markdown(
                        result["answer"]
                    )

                    citations = result.get(
                        "citations",
                        [],
                    )

                    if citations:

                        st.subheader("Sources")

                        for citation in citations:

                            st.markdown(
                                f"""
**{citation['document_id']}**

- Page: {citation['page_label']}
- Section: {citation['section_label']}
- Chunk: `{citation['chunk_id']}`
"""
                            )

                    latency = result.get(
                        "latency"
                    )

                    if latency:

                        with st.expander(
                            "Retrieval details"
                        ):

                            st.write(
                                f"Evidence chunks: "
                                f"{result['evidence_count']}"
                            )

                            st.write(
                                f"Evidence tokens: "
                                f"{result['evidence_tokens']}"
                            )

                            st.write(
                                f"Model: "
                                f"{result['model']}"
                            )

                            st.write(
                                f"Total latency: "
                                f"{latency['total_seconds']:.2f} s"
                            )

                else:

                    try:
                        error = response.json()

                        detail = error.get(
                            "detail",
                            "Request failed.",
                        )

                    except ValueError:
                        detail = (
                            "The backend returned "
                            "an unexpected error."
                        )

                    st.error(detail)