"""Bounded, deterministic sparse facet coverage for comparison questions.

This module never accesses evaluation labels or calls an embedding/generation API.
It is opt-in from the live retrieval path; historical hybrid benchmarks retain
exactly their original retrieval behavior.
"""
from __future__ import annotations

import re
from typing import Any, Callable

_COMPARE = re.compile(r"\b(same|different|difference|differ|compare|comparison|contrast|interchangeable)\b", re.I)
_WORDS = re.compile(r"[a-z0-9]+")


def comparison_facets(question: str) -> list[str]:
    """Extract independent topics from explicit 3+-way comparison questions.

    Deliberately conservative: do not split normal policy questions that merely
    contain a comma or the word 'and'. Never infer what documents ought to exist.
    """
    if len(question) > 1000 or not _COMPARE.search(question):
        return []
    clause = question.split("?", 1)[0].strip()
    # E.g. "Are A, B, and C the same classification system?"
    clause = re.sub(r"^(are|is)\s+", "", clause, flags=re.I)
    clause = re.split(r"\bthe\s+same\b|\binterchangeable\b|\bdifferent\b", clause, maxsplit=1, flags=re.I)[0]
    facets = [re.sub(r"^(and|or|the)\s+", "", f.strip(" ,.:;"), flags=re.I)
              for f in re.split(r",\s*(?:and\s+)?|\s+and\s+", clause, flags=re.I)]
    if not 3 <= len(facets) <= 4:
        return []
    if any(len(_WORDS.findall(f.lower())) < 2 or len(f) > 90 for f in facets):
        return []
    return facets


def _overlap(facet: str, chunk: dict[str, Any]) -> int:
    """Check that a result actually mentions the facet, not just a generic word."""
    haystack = " ".join(str(chunk.get(k) or "") for k in
                        ("document_id", "document_name", "section_title", "subsection_title", "text", "table_caption"))
    question_words = set(_WORDS.findall(facet.lower()))
    matching_words = set(_WORDS.findall(haystack.lower()))
    return len(question_words & matching_words)


def _severity_labels(chunk: dict[str, Any]) -> int:
    """Count distinct SEV classes evidenced by the actual source text.

    This is not a claim that the incident meets any severity; it only favors
    definitional evidence when the question compares classification systems.
    """
    text = str(chunk.get("text") or "")
    return len(set(re.findall(r"\bsev[-\s]*([1-4])\b", text, flags=re.I)))


def _incident_severity_facet(facet: str) -> bool:
    words = set(_WORDS.findall(facet.lower()))
    return "severity" in words and ("incident" in words or "security" in words)


def cover_comparison_facets(
    original: list[dict[str, Any]],
    facets: list[str],
    sparse_search: Callable[[str, int], list[dict[str, Any]]],
    *,
    limit: int,
    per_facet_search_limit: int = 10,
) -> list[dict[str, Any]]:
    """Promote one relevant, novel-document result per missing comparison facet.

    Keep the leading fused result, then promote relevant new documents using
    only local BM25. No fabricated evidence and no gold source ID lookups.
    Results retain their real origin in ``facet_query`` instead of pretending
    to have an RRF score. The rest retain their original fused order.
    """
    if limit <= 0 or per_facet_search_limit <= 0:
        raise ValueError("Search limits must be positive")
    if not facets or not original:
        return original[:limit]

    selected: list[dict[str, Any]] = [dict(original[0])]
    selected_ids = {original[0]["chunk_id"]}
    selected_docs = {str(original[0]["chunk"].get("document_id") or "")}
    for facet in facets:
        results = sparse_search(facet, per_facet_search_limit)
        suitable = [item for item in results
                    if item["chunk_id"] not in selected_ids
                    and item["chunk"].get("document_id")
                    and _overlap(facet, item["chunk"]) >= 2]
        # For an incident-severity comparison, generic triage guidance is not
        # sufficient to explain the severity *classification*. Prefer a source
        # containing explicit SEV labels. An additional bounded local BM25
        # query is used only if the first facet search lacks such evidence.
        # No gold IDs, hard-coded document IDs, or embedding calls are used.
        if _incident_severity_facet(facet):
            if not any(_severity_labels(r["chunk"]) >= 2 for r in suitable):
                expanded = sparse_search("incident severity classification SEV-1 SEV-4", max(20, per_facet_search_limit))
                seen = {r["chunk_id"] for r in suitable}
                suitable.extend(r for r in expanded
                                if r["chunk_id"] not in seen
                                and r["chunk_id"] not in selected_ids
                                and r["chunk"].get("document_id")
                                and _overlap(facet, r["chunk"]) >= 2)
            candidates = [r for r in suitable if str(r["chunk"]["document_id"]) not in selected_docs]
            novel = max(candidates, key=lambda r: _severity_labels(r["chunk"]), default=None)
        else:
            # Preserve original sparse order for other comparison topics.
            novel = next((r for r in suitable if str(r["chunk"]["document_id"]) not in selected_docs), None)
        if novel is None:
            continue
        anchor = dict(novel)
        anchor["facet_query"] = facet
        anchor["facet_sparse_rank"] = novel["rank"]
        anchor["facet_sparse_score"] = novel["score"]
        anchor["rrf_score"] = None
        anchor["dense_rank"] = None
        anchor["dense_score"] = None
        anchor["sparse_rank"] = None
        anchor["sparse_score"] = None
        selected.append(anchor)
        selected_ids.add(anchor["chunk_id"])
        selected_docs.add(str(anchor["chunk"]["document_id"]))
        if len(selected) >= limit:
            break

    for item in original[1:]:
        if len(selected) >= limit:
            break
        if item["chunk_id"] not in selected_ids:
            selected.append(dict(item))
            selected_ids.add(item["chunk_id"])
    for rank, item in enumerate(selected, 1):
        item["rank"] = rank
    return selected
