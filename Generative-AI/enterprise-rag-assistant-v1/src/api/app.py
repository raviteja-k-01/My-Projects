from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.genai import errors as genai_errors
from src.rag.rag_pipeline import EnterpriseRAGPipeline


# --------------------------------------------------
# Data expected from the client
# --------------------------------------------------

class AskRequest(BaseModel):
    question: str


# --------------------------------------------------
# RAG pipeline
# --------------------------------------------------

rag_pipeline: EnterpriseRAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):

    global rag_pipeline

    # Load the RAG system once when FastAPI starts
    rag_pipeline = EnterpriseRAGPipeline()

    yield

    # Close resources when FastAPI shuts down
    if rag_pipeline is not None:
        rag_pipeline.close()


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Enterprise RAG Assistant",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# --------------------------------------------------
# RAG endpoint
# --------------------------------------------------

@app.post("/ask")
def ask(request: AskRequest):

    if rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not ready.",
        )

    try:
        result = rag_pipeline.answer(
            question=request.question
        )

        return result
    except genai_errors.ClientError as exc:

        if getattr(exc, "code", None) == 429:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Embedding service quota is currently "
                    "exhausted. Please retry later."
                ),
            ) from exc

        raise HTTPException(
            status_code=502,
            detail="Embedding service request failed.",
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc