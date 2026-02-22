"""
backend/main.py
FastAPI backend — Multimodal AI Assistant
Models: llama-4-scout (vision) | llama-3.3-70b (text) | whisper-large-v3-turbo (audio)
All FREE on Groq.
"""

import os
import uuid
import logging
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.parsers import parse_file, SUPPORTED_EXTENSIONS
from utils.context import merge_contexts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
TEXT_MODEL   = "llama-3.3-70b-versatile"

app = FastAPI(
    title="Multimodal AI Assistant",
    description="PDF, DOCX, XLSX, Images, Audio, Video — powered by Groq.",
    version="2.0.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SESSIONS: dict = {}


def _get_groq():
    from groq import Groq
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


def _has_images(parsed_files):
    return any(
        f.get("parsed", {}).get("images") or
        f.get("parsed", {}).get("type") in ("image", "video")
        for f in parsed_files
    )


def _format_hint(fmt):
    return {
        "table":   "Format your answer as a markdown table.",
        "report":  "Format your answer as a structured report with headings.",
        "bullets": "Format your answer as bullet points.",
        "json":    "Return valid JSON only, no extra text.",
    }.get(fmt, "")


class AskRequest(BaseModel):
    session_id: str
    prompt: str
    output_format: str = "auto"

class AskResponse(BaseModel):
    session_id: str
    answer: str
    files_used: List[str]
    model_used: str
    tokens_used: int


@app.get("/")
def root():
    return {
        "status": "running",
        "version": "2.0.0",
        "supported": sorted(SUPPORTED_EXTENSIONS),
        "docs": "/docs"
    }


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    session_id = str(uuid.uuid4())
    parsed_files, errors = [], []

    for f in files:
        try:
            data = await f.read()
            parsed = parse_file(f.filename, data)
            parsed_files.append({"filename": f.filename, "parsed": parsed})
            logger.info(f"OK: {f.filename} ({parsed['type']}) -> {session_id}")
        except Exception as e:
            errors.append({"filename": f.filename, "error": str(e)})
            logger.error(f"FAIL: {f.filename} -> {e}")

    if not parsed_files:
        raise HTTPException(status_code=400, detail={"message": "No files could be parsed.", "errors": errors})

    SESSIONS[session_id] = parsed_files
    return {
        "session_id": session_id,
        "files_uploaded": [{"filename": p["filename"], "type": p["parsed"]["type"]} for p in parsed_files],
        "errors": errors
    }


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found. Upload files first.")

    parsed_files = SESSIONS[req.session_id]
    prompt = req.prompt
    hint = _format_hint(req.output_format)
    if hint:
        prompt += f"\n\n{hint}"

    use_vision = _has_images(parsed_files)
    model = VISION_MODEL if use_vision else TEXT_MODEL
    messages = merge_contexts(parsed_files, prompt, use_vision=use_vision)

    groq = _get_groq()

    try:
        resp = groq.chat.completions.create(
            model=model, messages=messages, max_tokens=4096, temperature=0.3
        )
        answer = resp.choices[0].message.content
        tokens = resp.usage.total_tokens
        logger.info(f"{req.session_id} | {model} | {tokens} tokens")

    except Exception as e:
        if use_vision:
            logger.warning(f"Vision failed, falling back to text: {e}")
            try:
                messages2 = merge_contexts(parsed_files, prompt, use_vision=False)
                resp = groq.chat.completions.create(
                    model=TEXT_MODEL, messages=messages2, max_tokens=4096, temperature=0.3
                )
                answer = resp.choices[0].message.content
                tokens = resp.usage.total_tokens
                model = TEXT_MODEL + " (fallback)"
            except Exception as e2:
                raise HTTPException(status_code=500, detail=f"Error: {str(e2)}")
        else:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    return AskResponse(
        session_id=req.session_id,
        answer=answer,
        files_used=[p["filename"] for p in parsed_files],
        model_used=model,
        tokens_used=tokens
    )


@app.get("/session/{session_id}")
def get_session(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")
    files = SESSIONS[session_id]
    return {
        "session_id": session_id,
        "file_count": len(files),
        "uses_vision": _has_images(files),
        "files": [{"filename": f["filename"], "type": f["parsed"]["type"]} for f in files]
    }


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")
    del SESSIONS[session_id]
    return {"message": "Session deleted."}
