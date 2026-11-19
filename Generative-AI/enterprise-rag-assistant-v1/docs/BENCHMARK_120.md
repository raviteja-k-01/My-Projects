# RAG-120 evaluation: separate from the original 30-question gold set

## Scope and provenance

`data/evaluation/rag_eval_120.jsonl` is the **original supplied dataset**, not a newly generated or automatically corrected benchmark. It contains 120 questions (80 `dev`, 40 `test`), 108 marked answerable and 12 marked unanswerable, and 355 expected-fact prompts. Its SHA-256 is `040520cd1c25a006216e345cff6001b696838c68522421a40205a205331b9086`.

The original 30-question evaluation and its checkpoint/review formats are **unchanged**. Never pass this 120-question dataset to the 30-question gold loader. The JSONL schemas differ.

## Offline commands (run from the RAG project directory)

```bash
# Read-only dataset structure (no API key or corpus required)
python -m scripts.evaluate_benchmark120 status

# Build a source-anchor audit and review worksheets. OUT must not exist.
python -m scripts.evaluate_benchmark120 audit \
  --chunks /absolute/path/to/processed/chunks.jsonl \
  --out /absolute/path/to/new-rag120-audit-directory

# Produce separate EMPTY human relevance labels. The file must not exist.
python -m scripts.evaluate_benchmark120 init-labels \
  --audit-dir /absolute/path/to/new-rag120-audit-directory \
  --out /absolute/path/to/rag120_labels_unreviewed.jsonl

# Once independent human-reviewed chunk labels AND real retrieval checkpoints exist:
python -m scripts.evaluate_benchmark120 score-retrieval \
  --labels /absolute/path/to/human-reviewed-labels.jsonl \
  --checkpoint /absolute/path/to/real-retrieval-checkpoint.jsonl \
  --split dev --k 5 --k 10
```

These operations require only local Python and files. **No embeddings, Gemini generation, Qdrant or network access.** Nothing is written inside `data/processed/` unless you choose that destination.

## Which checks are authoritative?

- **Located anchor:** An evidence snippet or composite table clause appears in a real corpus chunk, usually on a listed page. This is a *source-occurrence check*, not factual entailment. The current supplied corpus matched all 148 snippets; the audit still flags 15 missing *document-level* anchors across 14 answerable questions.
- **Expected facts:** All 355 require policy/source review. `fact_review.csv` is an empty worksheet; `candidate_chunk_ids_unverified` are search hints, **never gold labels**.
- **Unanswerable:** The 12 cases need source-wide qualified-negative review. The system cannot establish absence merely by finding no snippet. See `abstention_review.csv`.
- **Relevance labels:** `init-labels` creates unreviewed templates. A reviewer must explicitly set `review_state: "verified"`, `reviewer`, and `verified_relevant_chunk_ids` for answerable cases. No candidate chunk is pre-approved.
- **Retrieval scoring:** Only answerable questions with verified labels and a checkpoint are scored. Unanswerable cases are excluded from Recall/MRR, not counted as retrieval failures. Reports show the denominator and skipped cases. Running on the `test` split is explicit; do not tune the system using test results.
- **Answer quality:** Recall/MRR do not establish factual correctness, correct citations or abstention. Actual generation review is a separate evaluation stage.

## Retrieval checkpoint schema (offline scorer)

One line per evaluated question, with ordered chunk IDs returned by the actual retriever:

```json
{"question_id": "E001", "retrieved_chunk_ids": ["<actual-chunk-id-1>", "<actual-chunk-id-2>"]}
```

Do not create synthetic checkpoints merely to obtain a score. The scorer does not initiate retrieval or use model quota. It validates duplicate IDs and rejects malformed records.

## Engineering constraints

No hardcoded question-to-answer maps. No assignment of relevance from substring matches. No automatic edits to the user's source documents, gold facts, or review sign-offs. Outputs refuse overwrite. Source hashes are retained to detect stale audits/labels. The tool is standard-library-only; tests run under the project's offline pytest suite.

This integration is **an offline benchmark audit + review-label gate + future retrieval scorer**, not a claim that RAG-120 has been semantically validated or evaluated end-to-end.

## Strict-hybrid RAG-120 capture (budgeted and resumable)

The benchmark now supports an *optional* live strict-hybrid retrieval run on
the 80 development questions. It uses the existing `HybridRetriever.retrieve`
method (Gemini query embedding + Qdrant dense + BM25 sparse + RRF). **No
BM25 fallback** is permitted. The separate 40-question `test` split stays
locked until you explicitly opt in.

```bash
# Safe preview: no provider initialization, no network requests, no writes
python -m scripts.evaluate_benchmark120 capture-hybrid \
  --chunks /path/to/processed/chunks.jsonl \
  --checkpoint data/processed/benchmark120_hybrid_dev.jsonl

# When API quota is available AND you authorize ONE new query embedding:
python -m scripts.evaluate_benchmark120 capture-hybrid \
  --chunks /path/to/processed/chunks.jsonl \
  --checkpoint data/processed/benchmark120_hybrid_dev.jsonl \
  --max-new 1 --confirm-live

# Score checkpoint *only* against independently reviewed labels:
python -m scripts.evaluate_benchmark120 score-retrieval \
  --labels /path/to/reviewed_labels.jsonl \
  --checkpoint data/processed/benchmark120_hybrid_dev.jsonl \
  --split dev --k 5 --k 10
```

A checkpoint row is saved atomically after each successful query. On resume,
question text, dataset SHA, corpus SHA, retrieval configuration, chunk IDs
and split must all match; stale or incompatible data is rejected. Running
with no `--max-new` performs only a read-only preview. Retrieving questions
does **not** certify the labels, validate expected facts or generate answers.
Quota limits and provider availability remain external constraints.
