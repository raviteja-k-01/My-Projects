"""Inspect gold question evidence against the exact verified 1,985-chunk corpus.

No Gemini API calls, no Qdrant access and no writes. Displays the full text
of the manually selected gold chunks so a reviewer can validate each expected
fact *before* judging generated answers.

Example:
    python -m scripts.audit_gold_evidence \
        --chunks /mnt/c/Users/ravit/Downloads/rag-v1-retrieval-artifacts/processed/chunks.jsonl \
        --question-id Q001
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.build_dense_index import verify_corpus
from src.evaluation.loader import load_gold_questions


def validate_gold_references(
    gold: list[dict[str, Any]],
    chunks: dict[str, Any],
) -> dict[str, Any]:
    """Validate references structurally; this does NOT prove factual support."""
    if not gold:
        raise ValueError("Gold dataset is empty")
    if not chunks:
        raise ValueError("Corpus is empty")
    missing: list[tuple[str, str]] = []
    wrong_origin: list[str] = []
    reference_count = 0
    for row in gold:
        ids = row["expected_chunk_ids"]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate gold chunk IDs: {row['question_id']}")
        for chunk_id in ids:
            reference_count += 1
            if chunk_id not in chunks:
                missing.append((row["question_id"], chunk_id))
        # A cross-document question can have many other document IDs. Where
        # a primary document is specified, at least one ID must reference it.
        expected_document = row.get("expected_document_id")
        if ids and expected_document and all(
            chunk_id in chunks
            and chunks[chunk_id].document_id != expected_document
            for chunk_id in ids
        ):
            wrong_origin.append(row["question_id"])
    if missing or wrong_origin:
        raise ValueError(
            f"Unresolved gold references: missing={missing[:10]}, "
            f"wrong_primary_document={wrong_origin[:10]}"
        )
    return {
        "questions": len(gold),
        "retrieval_references": reference_count,
        "unique_referenced_chunks": len({
            item for row in gold for item in row["expected_chunk_ids"]
        }),
        "answerable": sum(row["answerable"] for row in gold),
        "unanswerable": sum(not row["answerable"] for row in gold),
        "expected_facts": sum(len(row["expected_facts"]) for row in gold),
        "question_types": dict(sorted(Counter(
            row["question_type"] for row in gold
        ).items())),
    }


def format_question(
    row: dict[str, Any], chunks: dict[str, Any]
) -> str:
    lines = [
        f"QUESTION: {row['question_id']}: {row['question']}",
        f"TYPE: {row['question_type']} | ANSWERABLE: {row['answerable']}",
        "",
        "GOLD EXPECTED FACTS — HUMAN VALIDATION REQUIRED:",
    ]
    for number, fact in enumerate(row["expected_facts"], start=1):
        lines.append(f"  {number}. {fact}")
    if not row["expected_chunk_ids"]:
        lines.append(
            "\nNO SELECTED GOLD CHUNKS. This is not a statement that "
            "the system must retrieve nothing."
        )
    for chunk_id in row["expected_chunk_ids"]:
        chunk = chunks[chunk_id]
        lines.extend((
            "",
            f"GOLD CHUNK: {chunk.chunk_id}",
            f"DOCUMENT: {chunk.document_id} | PAGES: "
            f"{chunk.page_start}-{chunk.page_end}",
            f"SECTION: {chunk.section_number or 'unspecified'} | "
            f"{chunk.section_title or 'unspecified'}",
            f"CONTENT TYPE: {chunk.content_type}",
            "EXACT CHUNK TEXT:",
            chunk.text,
        ))
    lines.append(
        "\nREVIEWER ACTION: For each expected fact, determine whether the "
        "actual source text supports it. Missing support is a gold-label "
        "problem, not an automatic model failure."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument("--question-id", type=str, default="Q001")
    args = parser.parse_args()
    indexed = verify_corpus(args.chunks)
    chunks = {item.chunk_id: item for item in indexed}
    if len(chunks) != len(indexed):
        raise ValueError("Duplicate corpus chunk IDs")
    gold = load_gold_questions()
    summary = validate_gold_references(gold, chunks)
    print(
        f"VERIFIED CORPUS: {len(chunks)} chunks | "
        f"GOLD: {summary['questions']} questions | "
        f"EXPECTED FACTS: {summary['expected_facts']} | "
        f"REFERENCES: {summary['retrieval_references']}"
    )
    print("REFERENCE CHECK: PASSED. FACTUAL SUPPORT: NOT AUTOMATICALLY VERIFIED.")
    selected = next(
        (row for row in gold if row["question_id"] == args.question_id), None
    )
    if selected is None:
        raise ValueError(f"Unknown gold question ID: {args.question_id}")
    print("\n" + format_question(selected, chunks))


if __name__ == "__main__":
    main()
