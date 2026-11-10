\# RBI Financial Regulations RAG Assistant



AI assistant that answers questions about RBI financial regulations using Retrieval-Augmented Generation (RAG).



\## Features



\- Hybrid retrieval (FAISS + BM25)

\- BGE-M3 embeddings

\- BGE reranker

\- Local LLM (Qwen2.5 via Ollama)

\- Streamlit UI

\- Source citations



\## Architecture



Documents

→ Chunking

→ BGE-M3 Embeddings

→ FAISS Vector DB

→ BM25 Search

→ Reranker

→ Qwen LLM

→ Answer + Sources



\## Setup



Install dependencies



pip install -r requirements.txt



Install Ollama



https://ollama.ai



Pull model



ollama pull qwen2.5:14b



Run application



streamlit run app/app.py



\## Example Questions



\- What is KYC according to RBI?

\- What are Basel III capital requirements?

\- What is priority sector lending?

