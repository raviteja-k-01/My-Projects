from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="Enterprise Knowledge Engine",
    description="Hybrid RAG System with BM25 + FAISS + Reranker + LLM",
    version="1.0"
)

app.include_router(router)