"""Conservative, zero-network routing for obvious non-policy requests.

This is not an exhaustive out-of-domain classifier. Ambiguous requests stay
on the RAG path rather than being rejected by an uncalibrated threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Any


WELCOME = (
    "Hi! I can help with HR, security, access, incidents, and other "
    "questions about the enterprise policy library. What would you like to know?"
)
OUT_OF_SCOPE = (
    "I can answer questions using the enterprise policy library, but not "
    "general knowledge or current-event questions. Try asking about HR, "
    "security, access, incidents, or workplace procedures."
)
NO_RELEVANT_EVIDENCE = (
    "I couldn't find relevant evidence in the available enterprise policies "
    "to answer that question. Try asking a more specific policy question."
)


@dataclass(frozen=True)
class QuickRoute:
    response_type: str
    answer: str


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


# Exact/complete greetings only. "Hi, how do I revoke access?" must not match.
_GREETINGS = frozenset({
    "hi", "hello", "hey", "hey there", "hello there", "hi there",
    "hi assistant", "hello assistant", "hey assistant",
    "how are you", "how are you doing", "hi how are you",
    "hi how are you doing", "hey how are you",
    "hello how are you", "good morning", "good afternoon", "good evening",
    "thank you", "thanks", "thanks a lot", "bye", "goodbye",
})

# Clear policy intent overrides category keywords: "What is our policy on
# election-related social media?" remains a possible enterprise question.
_POLICY_CUES = re.compile(
    r"\b(?:policy|policies|company|workplace|workforce|employee|employees|"
    r"worker|workers|contractor|contractors|manager|managers|hr|onboarding|"
    r"offboarding|privileged|access|authorization|security|incident|"
    r"incident response|sev[ -]?[0-9]|audit|compliance|risk|"
    r"client|project|vendor|termination|payroll|leave|benefits|"
    r"password|vpn|account|approval|approvals|control|controls|"
    r"remote work|people operations|provisioning|revocation|"
    r"training|identity|governance|procedure|procedures|"
    r"documents|documentation|internal|work from home|wfh|"
    r"remote|remotely|fired|layoff|layoffs|resignation|quit|"
    r"vacation|sick leave|sick day|laptop|device|equipment|"
    r"lost badge|badge|injury|harassment|promotion)\b",
    re.IGNORECASE,
)

# Only explicit categories that are clearly outside the sample policy library.
# Do not reject generic questions based solely on absence of policy keywords.
_CLEAR_EXTERNAL = (
    re.compile(r"\b(?:modi|fifa|world cup|ipl|olympics|"
               r"presidential election|elections?|prime minister|"
               r"president of (?:india|the united states|usa))\b"),
    re.compile(r"\b(?:weather|forecast|temperature)\b"),
    re.compile(r"\b(?:latest news|breaking news|stock price|bitcoin price)\b"),
    re.compile(r"\b(?:capital of [a-z]+|recipe for|how to cook)\b"),
    re.compile(r"\b(?:tell me (?:a |some )?jokes?|write (?:me )?a poem)\b"),
    re.compile(r"\b(?:write|generate|debug|fix)\b.{0,55}"
               r"\b(?:python|javascript|java|html|sql)\b.{0,35}"
               r"\b(?:code|script|program)\b"),
    re.compile(r"\b(?:python|javascript|java|html|sql)\b.{0,55}"
               r"\b(?:code|script|program)\b"),
)
_GREETING_PREFIX = re.compile(r"^(?:hi|hey|hello)(?: there| assistant)?\s+")

_WORDS = re.compile(r"[a-z][a-z0-9-]{2,}")
_STOPWORDS = frozenset({
    "the", "and", "for", "are", "was", "were", "what", "which",
    "who", "whom", "whose", "when", "where", "why", "how",
    "can", "could", "would", "should", "does", "did", "will",
    "that", "this", "these", "those", "from", "into", "with",
    "about", "there", "their", "them", "they", "you", "your",
    "our", "have", "has", "had", "isn", "is", "its", "some",
    "than", "then", "also", "just", "please", "tell", "know",
    "give", "explain", "describe", "many", "much", "any", "all",
    "need", "want", "process", "steps", "requirement", "requirements",
    "required", "rule", "rules", "work", "doing",
})


def quick_route(question: str) -> QuickRoute | None:
    """Handle only unmistakable chat and external-topic intents, locally."""
    normalized = _normalize(question)
    if normalized in _GREETINGS:
        return QuickRoute("greeting", WELCOME)

    without_hello = _GREETING_PREFIX.sub("", normalized)
    if not _POLICY_CUES.search(without_hello):
        if any(pattern.search(without_hello) for pattern in _CLEAR_EXTERNAL):
            return QuickRoute("out_of_scope", OUT_OF_SCOPE)
    return None


def lacks_relevant_evidence(question: str, evidence: list[dict[str, Any]]) -> bool:
    """Conservative lexical guard AFTER retrieval; no extra model requests.

    Reject only if the question has substantive words, no strong workplace
    cue, and not even one of those words appears in the selected evidence.
    Overlap does NOT establish factual support. Lack of overlap may be caused
    by paraphrases, so explicit policy/workplace questions are never rejected
    by this heuristic. This is not a calibrated semantic-relevance score.
    """
    if not evidence or _POLICY_CUES.search(question):
        return False
    keywords = set(_WORDS.findall(question.casefold())) - _STOPWORDS
    if not keywords:
        return False
    evidence_words = set()
    for item in evidence:
        evidence_words.update(_WORDS.findall(str(item.get("text", "")).casefold()))
    return keywords.isdisjoint(evidence_words)


def direct_response(
    question: str,
    *,
    response_type: str,
    answer: str,
    started: float,
    retrieval_mode: str = "not_run",
    retrieval_seconds: float | None = None,
    evaluation_trace: bool = False,
) -> dict[str, Any]:
    """Use the normal /ask response shape without claiming evidence or model use."""
    response: dict[str, Any] = {
        "question": question,
        "response_type": response_type,
        "retrieval_mode": retrieval_mode,
        "retrieval_source": "live" if retrieval_mode != "not_run" else "not_applicable",
        "retrieval_strategy": "none",
        "answer": answer,
        "citations": [],
        "citation_count": 0,
        "evidence_count": 0,
        "evidence_tokens": 0,
        "model": None,
        "generation_fallback_used": False,
        "usage": {},
        "latency": {
            "retrieval_seconds": retrieval_seconds,
            "reranking_seconds": 0.0,
            "generation_seconds": 0.0,
            "total_seconds": time.perf_counter() - started,
        },
    }
    if evaluation_trace:
        response["evaluation_trace"] = {
            "retrieved_chunk_ids": [],
            "selected_evidence": [],
        }
    return response
