"""Source-locked RAG-120 generation from *actual saved* strict-hybrid rankings.

Preview and report are offline. Capture requires explicit live generation consent,
reuses the production compose_answer path, and never instantiates a retriever.
No expected answer, source fact, or relevance label is supplied to the generator.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Callable

from scripts.evaluate_benchmark120 import load_dataset
from src.evaluation.benchmark120_evidence import read_jsonl, sha256
from src.evaluation.benchmark120_hybrid_capture import (
    _chunk_ids, _question_digest, _save_atomic, _verified_checkpoint,
)


def load_inputs(*, dataset_path: Path, corpus_path: Path,
                retrieval_path: Path, split: str = "dev", unlock_test: bool = False
                ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, str]]:
    """Verify all source hashes, question texts, rankings, IDs, split, and config."""
    if split not in {"dev", "test"}:
        raise ValueError("split must be dev or test")
    if split == "test" and not unlock_test:
        raise ValueError("Test split is held out; explicitly pass --unlock-test")
    if not retrieval_path.exists():
        raise ValueError("Missing saved retrieval checkpoint; do not repeat embeddings here")
    questions = [q for q in load_dataset(dataset_path) if q["split"] == split]
    if not questions:
        raise ValueError("Selected split has no questions")
    by_id = {q["question_id"]: q for q in questions}
    if len(by_id) != len(questions):
        raise ValueError("Repeated question ID")
    corpus_rows = read_jsonl(corpus_path)
    chunk_ids = _chunk_ids(corpus_path)
    chunks = {row["chunk_id"]: row for row in corpus_rows}
    if len(chunks) != len(chunk_ids):
        raise ValueError("Duplicate corpus chunk IDs")
    retrieval_rows = read_jsonl(retrieval_path)
    if not retrieval_rows:
        raise ValueError("Retrieval checkpoint is empty")
    limits = retrieval_rows[0].get("retrieval_limits")
    if (not isinstance(limits, dict) or set(limits) != {"dense_limit", "sparse_limit", "fused_limit"}
        or any(type(value) is not int or value < 1 for value in limits.values())):
        raise ValueError("Invalid retrieval limits in checkpoint")
    saved = _verified_checkpoint(
        retrieval_path, by_id, split, sha256(dataset_path), sha256(corpus_path), chunk_ids, limits
    )
    digests = {
        "dataset": sha256(dataset_path), "corpus": sha256(corpus_path),
        "retrieval": sha256(retrieval_path),
    }
    return questions, chunks, saved, digests


def _load_answers(path: Path, *, questions: list[dict[str, Any]], chunks: dict[str, Any],
                  retrieval: dict[str, Any], digests: dict[str, str]
                  ) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    known = {q["question_id"]: q for q in questions}
    saved: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        qid = row.get("question_id")
        if qid not in known or qid in saved:
            raise ValueError(f"Unknown or duplicate answer checkpoint: {qid}")
        q = known[qid]
        if (qid not in retrieval
            or row.get("split") != q["split"]
            or row.get("question_sha256") != _question_digest(q["question"])
            or row.get("source_dataset_sha256") != digests["dataset"]
            or row.get("source_corpus_sha256") != digests["corpus"]
            or row.get("source_retrieval_sha256") != digests["retrieval"]
            or row.get("retrieval_source") != "saved_hybrid_checkpoint"
            or row.get("retrieval_mode") != "hybrid"
            or not isinstance(row.get("answer"), str) or not row["answer"].strip()
            or not isinstance(row.get("model"), str) or not row["model"].strip()
            or not isinstance(row.get("selected_evidence"), list)
            or not isinstance(row.get("citations"), list)):
            raise ValueError(f"Stale or invalid saved answer: {qid}")
        selected_ids = [e.get("chunk_id") for e in row["selected_evidence"]]
        retrieved_ids = retrieval[qid]["retrieved_chunk_ids"]
        if (len(set(selected_ids)) != len(selected_ids)
            or any(cid not in retrieved_ids[:8] for cid in selected_ids)
            or any(e.get("evidence_id") != index for index, e in enumerate(row["selected_evidence"], 1))):
            raise ValueError(f"Answer evidence does not match retrieval context: {qid}")
        if (row.get("evidence_count") != len(selected_ids)
            or row.get("citation_count") != len(row["citations"])
            or any(e.get("text") != str(chunks[e["chunk_id"]].get("text", "")).strip()
                   or e.get("document_id") != chunks[e["chunk_id"]].get("document_id")
                   for e in row["selected_evidence"])):
            raise ValueError(f"Answer evidence text/metadata differs from locked corpus: {qid}")
        citation_ids = [c.get("evidence_id") for c in row["citations"]]
        if (len(set(citation_ids)) != len(citation_ids)
            or any(id not in range(1, len(selected_ids) + 1) for id in citation_ids)
            or any(c.get("chunk_id") != selected_ids[c["evidence_id"] - 1]
                   for c in row["citations"])):
            raise ValueError(f"Invalid citation evidence references: {qid}")
        saved[qid] = row
    return saved


def _candidates(retrieval_row: dict[str, Any], chunks: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"chunk_id": cid, "rank": rank, "chunk": chunks[cid]}
            for rank, cid in enumerate(retrieval_row["retrieved_chunk_ids"], start=1)]


def run(*, dataset_path: Path, corpus_path: Path, retrieval_path: Path,
        answers_path: Path, split: str = "dev", unlock_test: bool = False,
        max_new: int = 0, confirm_generation: bool = False, model_name: str | None = None,
        generator_factory: Callable[..., Any] | None = None,
        context_builder_factory: Callable[[], Any] | None = None,
        citation_builder_factory: Callable[[], Any] | None = None,
        compose_fn: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    """Offline preview by default. Explicit consent gates each bounded live batch."""
    if type(max_new) is not int or max_new < 0:
        raise ValueError("max_new must be a nonnegative integer")
    if max_new and not confirm_generation:
        raise ValueError("Generation requires --confirm-generation")
    if model_name is not None and (not isinstance(model_name, str) or not model_name.strip()):
        raise ValueError("model name must be nonempty")
    if answers_path.resolve() == retrieval_path.resolve():
        raise ValueError("Answer output and retrieval checkpoint must be separate")
    qs, chunks, retrieval, digests = load_inputs(
        dataset_path=dataset_path, corpus_path=corpus_path,
        retrieval_path=retrieval_path, split=split, unlock_test=unlock_test)
    saved = _load_answers(answers_path, questions=qs, chunks=chunks, retrieval=retrieval, digests=digests)
    pending = [q for q in qs if q["question_id"] in retrieval and q["question_id"] not in saved]
    report = {
        "split": split, "retrieval_checkpointed": len(retrieval),
        "questions": len(qs), "answer_checkpointed": len(saved),
        "answer_pending": len(pending), "newly_saved": 0,
        "live_generation_enabled": bool(max_new and confirm_generation),
        "new_embedding_calls": 0, "new_qdrant_queries": 0,
        "dataset_sha256": digests["dataset"], "corpus_sha256": digests["corpus"],
        "retrieval_checkpoint_sha256": digests["retrieval"],
        "model_names_captured": dict(sorted(Counter(r["model"] for r in saved.values()).items())),
        "answer_checkpoint": str(answers_path),
    }
    if not max_new or not pending:
        return report

    # Import only after the live gate. Never initialize HybridRetriever/Qdrant/embeddings.
    if generator_factory is None:
        from src.generation.answer_generator import GroundedAnswerGenerator
        generator_factory = GroundedAnswerGenerator
    if context_builder_factory is None:
        from src.retrieval.context_builder import ContextBuilder
        context_builder_factory = ContextBuilder
    if citation_builder_factory is None:
        from src.generation.citation_builder import CitationBuilder
        citation_builder_factory = CitationBuilder
    if compose_fn is None:
        from src.rag.answer_composer import compose_answer
        compose_fn = compose_answer

    generator = generator_factory(model_name=model_name)
    chosen_model = generator.model_name
    if any(r["model"] != chosen_model for r in saved.values()):
        raise ValueError("Different generation model in existing answer checkpoint; use a separate output file")
    context_builder = context_builder_factory()
    citations = citation_builder_factory()
    for q in pending[:max_new]:
        qid = q["question_id"]
        candidate_rows = _candidates(retrieval[qid], chunks)
        try:
            result = compose_fn(
                question=q["question"], candidates=candidate_rows,
                retrieval_mode="hybrid", retrieval_source="saved_hybrid_checkpoint",
                context_builder=context_builder, generator=generator,
                citation_builder=citations, retrieval_seconds=None,
                evaluation_trace=True,
            )
        except Exception as exc:
            # Never modify a successful earlier checkpoint when generation fails.
            raise RuntimeError(
                f"Generation stopped for {qid} ({type(exc).__name__}); previous answers preserved; "
                "no new query embeddings requested"
            ) from exc
        trace = result.get("evaluation_trace") or {}
        row = {
            "question_id": qid, "split": q["split"],
            "question_sha256": _question_digest(q["question"]),
            "source_dataset_sha256": digests["dataset"],
            "source_corpus_sha256": digests["corpus"],
            "source_retrieval_sha256": digests["retrieval"],
            "retrieval_source": result["retrieval_source"],
            "retrieval_mode": result["retrieval_mode"],
            "answer": result["answer"], "model": result["model"],
            "citations": result["citations"], "citation_count": result["citation_count"],
            "selected_evidence": trace.get("selected_evidence", []),
            "evidence_count": result["evidence_count"],
            "evidence_tokens": result["evidence_tokens"],
            "usage": result["usage"], "latency": result["latency"],
        }
        if row["model"] != chosen_model:
            raise ValueError("Generator response model name differs from configured model")
        saved[qid] = row
        _save_atomic(answers_path, (saved[record["question_id"]]
                                    for record in qs if record["question_id"] in saved))
        report["newly_saved"] += 1
    report["answer_checkpointed"] = len(saved)
    report["answer_pending"] = len(qs) - len(saved)
    report["model_names_captured"] = dict(sorted(Counter(r["model"] for r in saved.values()).items()))
    return report


def report(*, dataset_path: Path, corpus_path: Path, retrieval_path: Path,
           answers_path: Path, split: str = "dev", unlock_test: bool = False,
           review_export: Path | None = None) -> dict[str, Any]:
    qs, chunks, retrieval, digests = load_inputs(
        dataset_path=dataset_path, corpus_path=corpus_path, retrieval_path=retrieval_path,
        split=split, unlock_test=unlock_test)
    saved = _load_answers(answers_path, questions=qs, chunks=chunks, retrieval=retrieval, digests=digests)
    if review_export is not None:
        if review_export.exists():
            raise ValueError("Review export exists; refusing to overwrite human work")
        review_export.parent.mkdir(parents=True, exist_ok=True)
        with review_export.open("x", encoding="utf-8") as fh:
            for q in qs:
                row = saved.get(q["question_id"])
                if row is None:
                    continue
                fh.write(json.dumps({
                    "question_id": q["question_id"], "split": q["split"],
                    "question": q["question"], "answerable_in_dataset": q["answerable"],
                    "expected_facts_unverified": q["expected_facts"],
                    "generated_answer": row["answer"], "citations": row["citations"],
                    "selected_evidence": row["selected_evidence"],
                    "factual_completeness": "unreviewed", "groundedness": "unreviewed",
                    "citation_correctness": "unreviewed", "abstention_correctness": "unreviewed",
                    "review_notes": "",
                }, ensure_ascii=False) + "\n")
    return {
        "split": split, "questions": len(qs), "retrieval_checkpointed": len(retrieval),
        "answer_checkpointed": len(saved),
        "answers_with_no_citation_metadata": sum(not row["citations"] for row in saved.values()),
        "answerable_captured": sum(q["answerable"] and q["question_id"] in saved for q in qs),
        "unanswerable_captured": sum(not q["answerable"] and q["question_id"] in saved for q in qs),
        "human_factual_reviews_completed": 0,
        "warning": "Citation syntax and presence do not establish groundedness, correctness, or abstention quality.",
        "review_export": str(review_export) if review_export is not None else None,
    }
