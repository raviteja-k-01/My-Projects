"""Save current dev evidence selection for later offline before/after comparisons."""
import argparse
import json
from pathlib import Path

from src.evaluation.benchmark120_answer_replay import load_inputs, _candidates
from src.retrieval.context_builder import ContextBuilder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,default=Path("data/evaluation/rag_eval_120.jsonl"))
    parser.add_argument("--chunks",type=Path,required=True)
    parser.add_argument("--retrieval",type=Path,default=Path("data/processed/benchmark120_hybrid_dev.jsonl"))
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    qs,chunks,retrieval,digests = load_inputs(dataset_path=args.dataset,corpus_path=args.chunks,retrieval_path=args.retrieval)
    selections = {}
    for q in qs:
        qid = q["question_id"]
        if qid not in retrieval:
            raise ValueError("A complete dev retrieval checkpoint is required")
        context = ContextBuilder().build(_candidates(retrieval[qid],chunks))
        selections[qid] = [e["chunk_id"] for e in context["evidence"]]
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump({"source_hashes":digests,"selected_evidence":selections},stream,indent=2)


if __name__ == "__main__":
    main()
