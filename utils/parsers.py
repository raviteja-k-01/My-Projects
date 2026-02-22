"""
utils/parsers.py
Parses all file types: PDF, DOCX, XLSX, Images, Audio, Video
Windows-safe temp file handling throughout.
"""

import io
import os
import uuid
import base64
import tempfile
from pathlib import Path

import fitz
import pandas as pd
from docx import Document
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# ── Lazy Groq client (never created at import time) ──
_groq_client = None

def _get_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client


def _tmp_path(suffix: str) -> str:
    """Generate a unique temp file path (Windows-safe, no auto-delete)."""
    return os.path.join(tempfile.gettempdir(), f"mmai_{uuid.uuid4().hex}{suffix}")


def _safe_delete(path: str):
    """Delete a file, ignoring errors (Windows file lock safe)."""
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except Exception:
        pass


# ────────────────────────────────────────────
# PDF
# ────────────────────────────────────────────

def parse_pdf(file_bytes: bytes) -> dict:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages, images = [], []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append(f"[Page {page_num}]\n{text}")
        for i, img in enumerate(page.get_images(full=True)):
            base_img = doc.extract_image(img[0])
            images.append({
                "base64": base64.b64encode(base_img["image"]).decode(),
                "media_type": f"image/{base_img['ext']}",
                "page": page_num
            })

    return {
        "type": "pdf",
        "text": "\n\n".join(pages),
        "images": images,
        "page_count": len(doc)
    }


# ────────────────────────────────────────────
# DOCX
# ────────────────────────────────────────────

def parse_docx(file_bytes: bytes) -> dict:
    doc = Document(io.BytesIO(file_bytes))
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    tables = []
    for i, tbl in enumerate(doc.tables, 1):
        rows = [" | ".join(c.text.strip() for c in row.cells) for row in tbl.rows]
        tables.append(f"[Table {i}]\n" + "\n".join(rows))

    images = []
    for rel in doc.part.rels.values():
        if "image" in rel.reltype:
            images.append({
                "base64": base64.b64encode(rel.target_part.blob).decode(),
                "media_type": "image/png"
            })

    text = "\n\n".join(paras)
    if tables:
        text += "\n\n" + "\n\n".join(tables)

    return {"type": "docx", "text": text, "images": images}


# ────────────────────────────────────────────
# XLSX
# ────────────────────────────────────────────

def parse_xlsx(file_bytes: bytes) -> dict:
    xl = pd.ExcelFile(io.BytesIO(file_bytes))
    sheets = []
    for name in xl.sheet_names:
        df = xl.parse(name).fillna("")
        sheets.append(f"[Sheet: {name}]\n{df.to_string(index=False)}")
    return {
        "type": "xlsx",
        "text": "\n\n".join(sheets),
        "images": [],
        "sheet_count": len(xl.sheet_names)
    }


# ────────────────────────────────────────────
# Image
# ────────────────────────────────────────────

def parse_image(file_bytes: bytes, media_type: str) -> dict:
    img = Image.open(io.BytesIO(file_bytes))
    w, h = img.size
    return {
        "type": "image",
        "text": f"[Image: {w}x{h} pixels]",
        "images": [{"base64": base64.b64encode(file_bytes).decode(), "media_type": media_type}]
    }


# ────────────────────────────────────────────
# Audio — Groq Whisper
# ────────────────────────────────────────────

def parse_audio(file_bytes: bytes, filename: str) -> dict:
    ext = Path(filename).suffix.lower()
    tmp = _tmp_path(ext)

    try:
        # Write bytes to temp file
        with open(tmp, "wb") as f:
            f.write(file_bytes)

        # Transcribe
        with open(tmp, "rb") as f:
            result = _get_client().audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=(filename, f),
                response_format="verbose_json"
            )

        return {
            "type": "audio",
            "text": f"[Audio Transcription]\n{result.text.strip()}",
            "images": [],
            "duration_seconds": getattr(result, "duration", None),
            "language": getattr(result, "language", "unknown")
        }
    finally:
        _safe_delete(tmp)


# ────────────────────────────────────────────
# Video — Whisper transcription + cv2 frames
# ────────────────────────────────────────────

def parse_video(file_bytes: bytes, filename: str) -> dict:
    ext = Path(filename).suffix.lower()
    video_tmp = _tmp_path(ext)
    audio_tmp = _tmp_path(".mp3")
    transcript_text = ""
    images = []
    duration = 0

    try:
        # Write video to temp file
        with open(video_tmp, "wb") as f:
            f.write(file_bytes)

        # ── Step 1: Extract audio and transcribe ──
        try:
            try:
                from moviepy import VideoFileClip
            except ImportError:
                from moviepy.editor import VideoFileClip

            clip = VideoFileClip(video_tmp)
            duration = clip.duration

            if clip.audio is not None:
                clip.audio.write_audiofile(audio_tmp, logger=None)
                clip.close()

                with open(audio_tmp, "rb") as af:
                    audio_bytes = af.read()

                audio_result = parse_audio(audio_bytes, "video_audio.mp3")
                transcript_text = audio_result["text"]
            else:
                clip.close()
                transcript_text = "[No audio track found in video]"

        except Exception as e:
            transcript_text = f"[Audio extraction failed: {str(e)}]"

        finally:
            _safe_delete(audio_tmp)

        # ── Step 2: Extract key frames with OpenCV ──
        try:
            import cv2
            cap = cv2.VideoCapture(video_tmp)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            interval = max(int(fps * 30), 1)  # one frame every 30 seconds
            frame_idx = 0

            while cap.isOpened() and len(images) < 8:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % interval == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(rgb)
                    pil.thumbnail((640, 480))
                    buf = io.BytesIO()
                    pil.save(buf, format="JPEG", quality=75)
                    images.append({
                        "base64": base64.b64encode(buf.getvalue()).decode(),
                        "media_type": "image/jpeg",
                        "timestamp": f"{frame_idx / fps:.1f}s"
                    })
                frame_idx += 1

            cap.release()

        except Exception as e:
            pass  # frames optional

    finally:
        _safe_delete(video_tmp)

    summary_lines = [f"[Video: {filename}]"]
    if duration:
        summary_lines.append(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")
    if images:
        summary_lines.append(f"Key frames extracted: {len(images)}")
    summary_lines.append(transcript_text)

    return {
        "type": "video",
        "text": "\n".join(summary_lines),
        "images": images,
        "duration_seconds": duration,
        "frame_count": len(images)
    }


# ────────────────────────────────────────────
# Router
# ────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xls",
    ".png", ".jpg", ".jpeg", ".webp", ".gif",
    ".mp3", ".wav", ".m4a", ".ogg", ".flac",
    ".mp4", ".avi", ".mov", ".mkv", ".webm"
}

_IMAGE_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"
}
_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_file(filename: str, file_bytes: bytes) -> dict:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return parse_pdf(file_bytes)
    elif ext in (".docx", ".doc"):
        return parse_docx(file_bytes)
    elif ext in (".xlsx", ".xls"):
        return parse_xlsx(file_bytes)
    elif ext in _IMAGE_TYPES:
        return parse_image(file_bytes, _IMAGE_TYPES[ext])
    elif ext in _AUDIO_EXTS:
        return parse_audio(file_bytes, filename)
    elif ext in _VIDEO_EXTS:
        return parse_video(file_bytes, filename)
    else:
        raise ValueError(f"Unsupported file type: '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
