"""Offline development-only evidence comparison. Anchor matches are NOT gold recall.

Reads a saved before-change selection snapshot and source-locked retrieval.
Never calls a provider, edits a label, or evaluates held-out question content.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

from src.evaluation.benchmark120_answer_replay import _candidates, load_inputs
from src.evaluation.benchmark120_evidence import Chunk, assess_snippet, snippet_list
from src.retrieval.context_builder import ContextBuilder


def compare(dataset: Path, corpus: Path, retrieval: Path, baseline: Path) -> dict:
    qs, chunks, rankings, digests = load_inputs(dataset_path=dataset, corpus_path=corpus,
                                              retrieval_path=retrieval, split="dev")
    before = json.loads(baseline.read_text(encoding="utf-8"))
    if before["source_hashes"] != digests:
        raise ValueError("Baseline inputs differ from current inputs")
    source_chunks = {cid: Chunk.from_row(c) for cid, c in chunks.items()}
    def coverage(q, ids):
        by_doc = defaultdict(list)
        for cid in ids:
            c = source_chunks[cid]
            by_doc[c.document_id].append(c)
        snippets = snippet_list(q["evidence"])
        matches = [assess_snippet(q, snippet, by_doc) for snippet in snippets]
        return {"matched": sum(bool(m["chunk_ids"]) for m in matches),
                "anchors": len(matches), "details": matches}
    rows = []
    for q in qs:
        qid = q["question_id"]
        old = before["selected_evidence"][qid]
        if any(cid not in rankings[qid]["retrieved_chunk_ids"] for cid in old):
            raise ValueError("Baseline selection does not belong to the locked ranking")
        context = ContextBuilder().build(_candidates(rankings[qid], chunks))
        new = [e["chunk_id"] for e in context["evidence"]]
        rows.append({"question_id": qid, "before": old, "after": new,
                     "before_anchors": coverage(q, old), "after_anchors": coverage(q, new),
                     "estimated_body_tokens": context["total_tokens"]})
    def total(phase, field):
        return sum(r[phase+"_anchors"][field] for r in rows)
    return {
        "split":"dev", "questions":len(qs), "source_hashes":digests,
        "changed_selections":sum(r["before"] != r["after"] for r in rows),
        "anchor_count":total("before","anchors"),
        "before_matched_anchors":total("before","matched"),
        "after_matched_anchors":total("after","matched"),
        "regressed_questions":[r["question_id"] for r in rows if r["after_anchors"]["matched"] < r["before_anchors"]["matched"]],
        "improved_questions":[r["question_id"] for r in rows if r["after_anchors"]["matched"] > r["before_anchors"]["matched"]],
        "before_mean_evidence_count":mean(len(r["before"]) for r in rows),
        "after_mean_evidence_count":mean(len(r["after"]) for r in rows),
        "retrieval_rankings_changed":False,
        "warning":"Literal source-anchor diagnostics only. No human-verified relevance labels; no answer-quality or recall improvement claimed.",
        "per_question":rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,default=Path("data/evaluation/rag_eval_120.jsonl"))
    p.add_argument("--chunks",type=Path,required=True)
    p.add_argument("--retrieval",type=Path,default=Path("data/processed/benchmark120_hybrid_dev.jsonl"))
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    result = compare(args.dataset,args.chunks,args.retrieval,args.baseline)
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k != "per_question"},indent=2))


if __name__ == "__main__":
    main()
