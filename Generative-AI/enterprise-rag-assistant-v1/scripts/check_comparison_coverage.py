"""Offline Q030-style comparison diagnostic using existing BM25 and saved rankings.

No Gemini embeddings, no Gemini generation, and no Qdrant requests. Does not
modify any evaluation checkpoint or corpus file.

Example:
  python -m scripts.check_comparison_coverage \\
    --chunks /mnt/c/Users/ravit/Downloads/rag-v1-retrieval-artifacts/processed/chunks.jsonl \\
    --question-id Q030 \\
    --require-docs AI-GOV-05 PROC-TPRM-11 SEC-IR-004
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.retrieval.comparison_coverage import comparison_facets, cover_comparison_facets
from src.retrieval.context_builder import ContextBuilder


def _load_case(path: Path, question_id: str) -> dict:
    found = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("question_id") == question_id:
            if found is not None:
                raise ValueError(f"Duplicate question ID: {question_id}")
            found = item
    if found is None:
        raise ValueError(f"Question ID absent from checkpoint: {question_id}")
    if not isinstance(found.get("retrieved_chunk_ids"), list):
        raise ValueError("Checkpoint lacks retrieved_chunk_ids")
    if not isinstance(found.get("question"), str):
        raise ValueError("Checkpoint lacks question text")
    return found


def _load_chunks(path: Path, wanted: set[str]) -> dict[str, dict]:
    results = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            chunk = json.loads(line)
            cid = chunk.get("chunk_id")
            if cid in wanted:
                if cid in results:
                    raise ValueError(f"Duplicate chunk ID: {cid}")
                results[cid] = chunk
    missing = wanted - set(results)
    if missing:
        raise ValueError(f"Saved chunk IDs missing from corpus: {sorted(missing)[:5]}")
    return results


def _docs(evidence: dict) -> list[str]:
    return list(dict.fromkeys(str(item["document_id"]) for item in evidence["evidence"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("data/processed/hybrid_retrieval_live.jsonl"))
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--question-id", default="Q030")
    parser.add_argument("--require-docs", nargs="*", default=[])
    args = parser.parse_args()

    case = _load_case(args.checkpoint, args.question_id)
    ranks = case["retrieved_chunk_ids"]
    chunks = _load_chunks(args.chunks, set(ranks))
    baseline = [
        {"chunk_id": cid, "rank": rank, "chunk": chunks[cid]}
        for rank, cid in enumerate(ranks, 1)
    ]
    facets = comparison_facets(case["question"])
    if not facets:
        raise ValueError("This question does not match conservative multi-facet comparison parsing")

    # Import only here: the pure parsing/selection tests do not require bm25s.
    from src.indexing.bm25_index import BM25Index
    bm25 = BM25Index()
    bm25.load()
    def sparse_search(facet: str, limit: int) -> list[dict]:
        return [{"chunk_id": row["chunk"]["chunk_id"], **row}
                for row in bm25.search(query=facet, limit=limit)]

    improved = cover_comparison_facets(
        original=baseline, facets=facets, sparse_search=sparse_search, limit=len(ranks)
    )
    context_builder = ContextBuilder(max_chunks=8, max_context_tokens=4000)
    before = context_builder.build(baseline)
    after = context_builder.build(improved)
    print(f"QUESTION: {args.question_id}")
    print(f"FACETS: {facets}")
    print(f"SAVED TOP-8 DOCUMENTS: {_docs(before)}")
    print(f"FACET-AWARE TOP-8 DOCUMENTS: {_docs(after)}")
    print(f"FACET ANCHORS: {[(r['facet_query'], r['chunk_id']) for r in improved if 'facet_query' in r]}")
    print("No embedding, generation, or Qdrant calls performed.")
    missing = sorted(set(args.require_docs) - set(_docs(after)))
    if missing:
        raise SystemExit(f"CHECK FAILED: required documents absent from selected evidence: {missing}")
    print("CHECK PASSED: requested document presence in selected evidence; semantic correctness NOT tested.")


if __name__ == "__main__":
    main()
