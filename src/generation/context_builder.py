from typing import List, Dict


def build_prompt(query: str, chunks: List[Dict]) -> List[Dict]:

    # Build context
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(f"[{i+1}] {chunk['text']}")

    context = "\n\n".join(context_parts)

    q = query.lower()

    # -------------------------
    # Dynamic instruction routing
    # -------------------------
    if "difference" in q:
        instruction = (
            "Separate the answer into clearly labeled categories.\n"
            "Do NOT mix items between categories.\n"
        )
    elif any(x in q for x in ["require", "required", "requirements", "list"]):
        instruction = (
            "State the outcome or consequence FIRST.\n"
            "Then include supporting rule if needed.\n"
        )
    elif "require" in q or "list" in q:
        instruction = (
            "Extract ALL required items mentioned in the context.\n"
            "Do NOT omit any item.\n"
            "Return as a complete list.\n"
        )
    elif "who" in q or "approval" in q:
        instruction = (
            "Clearly identify the responsible person or authority.\n"
            "Mention if conditions apply.\n"
        )
    else:
        instruction = (
            "Extract all relevant information completely.\n"
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an enterprise policy assistant.\n\n"
                "Strict Rules:\n"
                "- Answer ONLY using the provided context.\n"
                "- Do NOT use outside knowledge.\n"
                "- Do NOT hallucinate.\n"
                "- If information is missing, say: Not found in provided context.\n\n"

                "Extraction Rules:\n"
                "- Do NOT summarize unless asked.\n"
                "- Extract ALL relevant details.\n"
                "- Do NOT omit items if multiple exist.\n"
                "- Do NOT merge different rules.\n"
                "- Keep answers structured.\n"
            )
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{query}\n\n"
                f"Context:\n{context}\n\n"
                f"Instructions:\n{instruction}\n\n"
                "Answer (use bullet points or structured format):"
            )
        }
    ]

    return messages


def extract_citations(chunks: List[Dict]) -> List[Dict]:
    seen = set()
    citations = []

    for c in chunks:
        key = (c.get("doc_id"), c.get("page"))

        if key not in seen:
            seen.add(key)
            citations.append({
                "doc_id": c.get("doc_id"),
                "page": c.get("page"),
                "section": c.get("section")
            })

    return citations