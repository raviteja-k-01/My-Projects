"""
utils/context.py
Builds Groq-compatible message payloads from parsed file content.
Auto-switches between vision and text-only format.
"""

from typing import List, Dict, Any

MAX_TEXT_CHARS = 24_000


def merge_contexts(parsed_files: List[Dict[str, Any]], user_prompt: str, use_vision: bool = True) -> List[Dict]:
    system = {
        "role": "system",
        "content": (
            "You are an advanced multimodal AI assistant. "
            "You can understand PDFs, Word documents, Excel sheets, images, "
            "audio transcriptions, and video content (frames + transcription). "
            "Use ALL provided content to answer accurately. "
            "Reference source files and timestamps where relevant. "
            "Format responses clearly using markdown."
        )
    }

    if use_vision:
        return [system, {"role": "user", "content": _vision_blocks(parsed_files, user_prompt)}]
    else:
        return [system, {"role": "user", "content": _text_only(parsed_files, user_prompt)}]


def _vision_blocks(parsed_files, user_prompt):
    blocks = []

    for f in parsed_files:
        fname = f.get("filename", "unknown")
        parsed = f.get("parsed", {})
        ftype = parsed.get("type", "unknown")

        # Header
        header = f"--- FILE: {fname} ({ftype.upper()}) ---"
        if ftype == "pdf":
            header += f"\nPages: {parsed.get('page_count', '?')}"
        elif ftype == "xlsx":
            header += f"\nSheets: {parsed.get('sheet_count', '?')}"
        elif ftype == "audio":
            d = parsed.get("duration_seconds")
            if d:
                header += f"\nDuration: {d:.1f}s | Language: {parsed.get('language', '')}"
        elif ftype == "video":
            d = parsed.get("duration_seconds")
            fc = parsed.get("frame_count", 0)
            if d:
                header += f"\nDuration: {d:.1f}s | Frames extracted: {fc}"

        blocks.append({"type": "text", "text": header})

        # Text content
        text = parsed.get("text", "").strip()
        blocks.append({"type": "text", "text": text[:MAX_TEXT_CHARS] if text else "[No text content]"})

        # Images / frames
        for img in parsed.get("images", []):
            if img.get("timestamp"):
                blocks.append({"type": "text", "text": f"[Frame at {img['timestamp']}]"})
            blocks.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{img['media_type']};base64,{img['base64']}",
                    "detail": "auto"
                }
            })

    blocks.append({"type": "text", "text": f"\n--- USER REQUEST ---\n{user_prompt}"})
    return blocks


def _text_only(parsed_files, user_prompt):
    parts = []
    for f in parsed_files:
        fname = f.get("filename", "unknown")
        parsed = f.get("parsed", {})
        ftype = parsed.get("type", "unknown")
        text = parsed.get("text", "").strip()
        img_count = len(parsed.get("images", []))
        block = f"--- FILE: {fname} ({ftype.upper()}) ---\n"
        if img_count:
            block += f"[Contains {img_count} image(s)]\n"
        block += text[:MAX_TEXT_CHARS] if text else "[No text content]"
        parts.append(block)
    return "\n\n".join(parts) + f"\n\n--- USER REQUEST ---\n{user_prompt}"
