import time
from typing import List, Dict, Any

import litellm

litellm.suppress_debug_info = True

from .context_builder import build_prompt


def extract_citations(chunks: List[Dict]) -> List[Dict]:
    seen = set()
    citations = []

    for c in chunks:
        key = (c.get("doc_id"), c.get("page"))

        if key in seen:
            continue

        seen.add(key)

        citations.append({
            "number": len(citations) + 1,
            "doc_id": c.get("doc_id", "unknown"),
            "page_num": c.get("page", 0),
            "department": c.get("department", "unknown"),
            "doc_type": c.get("doc_type", "unknown"),
            "version": c.get("version", "v1"),
            "preview": c.get("text", "")[:200]
        })

    return citations


def generate_answer(
    query: str,
    chunks: List[Dict[str, Any]],
    cfg: dict,
) -> Dict[str, Any]:
    """
    Generate answer using LLM
    """

    if not chunks:
        return {
            "answer": "Not found in provided documents.",
            "citations": [],
            "confidence": "low",
            "latency_ms": 0,
            "chunk_count": 0,
        }
    for c in chunks:
         c["source"] = f"{c.get('doc_id')} | Page {c.get('page')}"
    messages = build_prompt(query, chunks)

    start = time.time()

    try:
        response = litellm.completion(
            model=cfg["llm"]["model"],
            messages=messages,
            max_tokens=cfg["llm"]["max_tokens"],
            temperature=cfg["llm"]["temperature"],
        )

        answer = response.choices[0].message.content.strip()

        # Basic validation
        if not answer or len(answer.strip()) < 5:
                answer = "Not found in provided documents."

        # Optional formatting cleanup
        answer = answer.strip()



    except Exception as e:
        print(f"[GEN] Error: {e}")

        return {
            "answer": "Error generating response. Please try again.",
            "citations": [],
            "confidence": "low",
            "latency_ms": 0,
            "chunk_count": len(chunks),
        }

    latency_ms = int((time.time() - start) * 1000)

    if "not found" in answer.lower():
         confidence = "low"
    elif len(chunks) >= 4:
         confidence = "high"
    else:
         confidence = "medium"
    

    return {
        "answer": answer,
        "citations": extract_citations(chunks),
        "confidence": confidence,
        "latency_ms": latency_ms,
        "chunk_count": len(chunks),
    }