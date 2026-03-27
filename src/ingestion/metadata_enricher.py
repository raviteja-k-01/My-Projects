from typing import List, Dict


# 🔥 MUST match your file names (lowercase, underscores)
DOC_REGISTRY: Dict[str, Dict[str, str]] = {
    "code_of_conduct": {
        "department": "HR",
        "doc_type": "policy",
        "category": "conduct",
        "access_level": "all",
    },
    "data_privacy_policy": {
        "department": "Legal",
        "doc_type": "policy",
        "category": "privacy",
        "access_level": "all",
    },
    "employee_handbook": {
        "department": "HR",
        "doc_type": "handbook",
        "category": "employment",
        "access_level": "all",
    },
    "it_security_policy": {
        "department": "IT",
        "doc_type": "policy",
        "category": "security",
        "access_level": "all",
    },
    "leave_attendence_policy": {
        "department": "HR",
        "doc_type": "policy",
        "category": "leave",
        "access_level": "all",
    },
    "onboarding": {
        "department": "HR",
        "doc_type": "sop",
        "category": "onboarding",
        "access_level": "all",
    },
    "performance_management_policy": {
        "department": "HR",
        "doc_type": "policy",
        "category": "performance",
        "access_level": "all",
    },
    "travel_and_expenses_policy": {
        "department": "Finance",
        "doc_type": "policy",
        "category": "travel",
        "access_level": "all",
    },
}


DEFAULT_META = {
    "department": "general",
    "doc_type": "document",
    "category": "general",
    "access_level": "all",
}


def enrich_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    Attach metadata to each chunk based on doc_id.
    """

    enriched = []
    warned = set()

    for chunk in chunks:
        doc_id = chunk["doc_id"]

        meta = DOC_REGISTRY.get(doc_id)

        if not meta:
            if doc_id not in warned:
                print(f"[WARN] Missing metadata for: {doc_id}")
                warned.add(doc_id)
            meta = DEFAULT_META

        chunk.update(meta)
        enriched.append(chunk)

    print(f"[ENRICH] {len(enriched)} chunks enriched\n")

    return enriched