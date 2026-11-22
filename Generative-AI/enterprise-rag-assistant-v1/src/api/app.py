from contextlib import asynccontextmanager
import hmac
import logging
import os

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator
from google.genai import errors as genai_errors
from src.rag.rag_pipeline import EnterpriseRAGPipeline
from src.api.rate_limit import SlidingWindowRateLimiter
from src.generation.reliability import GenerationError


# --------------------------------------------------
# Data expected from the client
# --------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    evaluation_trace: bool = False

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
# Public demo limits are process-local safeguards, NOT provider quota guarantees.
# A single replica prevents per-process policy from being multiplied by workers.
demo_hourly_limiter = SlidingWindowRateLimiter(
    limit=int(os.getenv("RAG_DEMO_MAX_REQUESTS_PER_HOUR", "12")), window_seconds=3600
)
demo_daily_limiter = SlidingWindowRateLimiter(
    limit=int(os.getenv("RAG_DEMO_MAX_REQUESTS_PER_DAY", "24")), window_seconds=86400
)
logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):

    global rag_pipeline

    public_auth_required = os.getenv("RAG_REQUIRE_DEMO_AUTH", "").strip().lower() in {"1", "true", "yes"}
    if public_auth_required:
        demo_mode = os.getenv("RAG_DEMO_MODE", "").strip().lower() in {"1", "true", "yes"}
        token = os.getenv("RAG_DEMO_API_TOKEN", "")
        if not demo_mode or len(token) < 24:
            raise RuntimeError("Public deployment requires demo authentication.")

    # Load the RAG system once when FastAPI starts
    rag_pipeline = EnterpriseRAGPipeline()

    try:
        yield
    finally:
        # Also release resources when lifespan exits through an exception.
        pipeline, rag_pipeline = rag_pipeline, None
        if pipeline is not None:
            pipeline.close()


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


@app.get("/livez")
def livez():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not ready.")
    try:
        bm25 = getattr(rag_pipeline.retriever, "bm25", None)
        corpus = getattr(getattr(bm25, "retriever", None), "corpus", None)
        if not corpus:
            raise HTTPException(status_code=503, detail="RAG retrieval index is not ready.")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=503, detail="RAG pipeline readiness check failed.") from None
    return {"status": "ready"}


# --------------------------------------------------
# RAG endpoint
# --------------------------------------------------

# Only bounded diagnostic categories appear in logs, never raw provider
# responses, user questions, evidence text or exception details.
def classify_validation_error(exc: ValueError) -> str:
    message = str(exc)
    if message.startswith("Answer cites Evidence ") and "evidence does not exist" in message:
        return "citation_unknown_evidence"
    if message == "Malformed or unbracketed evidence reference":
        return "citation_malformed"
    if message == "Duplicate evidence IDs":
        return "citation_duplicate_ids"
    if message == "Answer cannot be empty":
        return "answer_empty"
    if message == "Evidence cannot be empty":
        return "evidence_empty"
    return "other_validation_error"


def enforce_rate_limit(http_request: Request) -> None:
    demo_mode = os.getenv("RAG_DEMO_MODE", "").strip().lower() in {"1", "true", "yes"}
    if demo_mode:
        required = os.getenv("RAG_DEMO_API_TOKEN", "")
        supplied = http_request.headers.get("X-RAG-Demo-Token", "")
        if len(required) < 24 or not hmac.compare_digest(supplied, required):
            raise HTTPException(status_code=401, detail="Demo authentication required.")
        # Fail closed before spending an embedding or generation request.
        if not demo_daily_limiter.allow("demo_global"):
            raise HTTPException(
                status_code=429, detail="Daily demo request limit reached.",
                headers={"Retry-After": "3600"},
            )
        if not demo_hourly_limiter.allow("demo_global"):
            raise HTTPException(
                status_code=429, detail="Hourly demo request limit reached.",
                headers={"Retry-After": "300"},
            )
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
def ask(
    request: AskRequest,
    http_response: Response,
    _rate_limit: None = Depends(enforce_rate_limit),
):

    if rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not ready.",
        )

    evaluation_trace = bool(request.evaluation_trace)
    if evaluation_trace and os.getenv("RAG_ALLOW_EVALUATION_TRACE", "").strip().lower() not in {
        "1", "true", "yes"
    }:
        raise HTTPException(
            status_code=403,
            detail="Evaluation trace is disabled.",
        )

    try:
        if evaluation_trace:
            result = rag_pipeline.answer(
                question=request.question,
                evaluation_trace=True,
            )
        else:
            result = rag_pipeline.answer(
                question=request.question,
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
    except GenerationError as exc:
        if exc.category == "quota" and exc.code == 429:
            if exc.retry_after_seconds is not None:
                http_response.headers["Retry-After"] = str(max(1, int(exc.retry_after_seconds)))
            safe_limits = (
                ("X-Ratelimit-Remaining-Requests", exc.ratelimit_remaining_requests),
                ("X-Ratelimit-Remaining-Tokens", exc.ratelimit_remaining_tokens),
                ("X-Ratelimit-Reset-Requests", exc.ratelimit_reset_requests),
                ("X-Ratelimit-Reset-Tokens", exc.ratelimit_reset_tokens),
            )
            for name, value in safe_limits:
                if value:
                    http_response.headers[name] = value[:64]
        raise HTTPException(
            status_code=502 if exc.category in {"invalid_response", "request", "authentication"} else 503,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail="RAG service is temporarily unavailable. Please retry later.",
        ) from exc

    except ValueError as exc:
        logger.warning("RAG_VALIDATION_FAILED category=%s", classify_validation_error(exc))
        raise HTTPException(
            status_code=502,
            detail="The RAG service could not produce a valid grounded answer.",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="RAG upstream service failed.") from exc
