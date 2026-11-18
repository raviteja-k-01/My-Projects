from contextlib import asynccontextmanager
import os

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from google.genai import errors as genai_errors
from src.rag.rag_pipeline import EnterpriseRAGPipeline
from src.api.rate_limit import SlidingWindowRateLimiter


# --------------------------------------------------
# Data expected from the client
# --------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)

    @field_validator("question")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("Question cannot be blank")
        return question


# --------------------------------------------------
# RAG pipeline
# --------------------------------------------------

rag_pipeline: EnterpriseRAGPipeline | None = None
rate_limiter = SlidingWindowRateLimiter(limit=int(os.getenv("RAG_RATE_LIMIT_PER_MINUTE", "12")))


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

def enforce_rate_limit(http_request: Request) -> None:
    # Use the actual peer address; forwarded headers are trusted only if the
    # deployment proxy and Uvicorn are independently configured to trust them.
    client_id = http_request.client.host if http_request.client else "unknown"
    if not rate_limiter.allow(client_id):
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please retry later.",
            headers={"Retry-After": "60"},
        )


@app.post("/ask")
def ask(request: AskRequest, _rate_limit: None = Depends(enforce_rate_limit)):

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