# Engineering audit and execution record — 2026-11-19

This is an evidence record, not a claim of release readiness.

Latest continuation: [offline engineering review](OFFLINE_REVIEW.md), covering
semantic-abstention limits, application tests, diff review and legacy retirement.
The original phase results below remain a historical record.

## Baseline

- Branch: `rag-v1/reproducible-retrieval-artifacts`; starting commit `37f523d`.
- Pre-existing work: `.gitignore` adds `.venv/`; untracked `backups/` is preserved.
- No applicable on-disk `AGENTS.md` was found in the project or its ancestors.
- Original WSL environment: Python 3.12.3, `~/.venvs/enterprise-rag-v1`.
  Installed direct libraries inspected match the pinned requirements.
- Baseline: `python -m pytest -q` — **77 passed in 5.06 seconds**.
- Gold: 30 questions, 27 answerable, 3 unanswerable, 121 expected facts.
- Q001–Q005 each have 20 saved ranked chunks and a captured answer.
  Q006–Q030 have neither. No answer has a completed human review.
- Q026, Q027, Q028 are unanswerable. Only Q027 lacks expected chunks.
- The original backup `backups/answer_eval_before_fix_20261118_184330.tar.gz`
  is readable; all 32 archived files match current files byte-for-byte.
- Repository-local `data/processed/chunks.jsonl` is absent. The external file
  `/mnt/c/Users/ravit/Downloads/rag-v1-retrieval-artifacts/processed/chunks.jsonl`
  exists and passes the indexed-build SHA-256 and 1,985-chunk validation.
  No corpus was substituted and no index was rebuilt.
- Initial protected hashes and parsed captures/checkpoints were recorded in
  WSL temporary storage, then reconstructed from the byte-identical original
  archive into durable `data/processed/engineering_baseline.json` after the
  temporary file was unavailable. All 33 protected files were reverified.

| Protected file | Initial SHA-256 |
| --- | --- |
| Gold questions | `697176fd654bf325c947d8e8c0133be87cebdbb266bb579ed7ae8e8d0e35501b` |
| Answer cases | `888a4b75cf4621905797aab6d7b099641d6a9fd3c2b6a1cdb1576e979b1a952f` |
| Hybrid checkpoint | `f3ddd189c98227930a5fad4f585b0f851e13887cf2df51d14a9594c8d1192b0e` |
| Exact indexed corpus | `41b7f8ff1d5db169a7cf5c945dbcc5d655578b6e25bea665c10dc7888a83abf4` |

## Working components and limits of the evidence

Source inspection and the existing offline suite cover ingestion ordering,
embedding shape/cache, safe dense-index resume, Qdrant backup validation,
BM25/dense RRF, context selection, citation mapping, API validation/rate limits,
strict retrieval, atomic evaluation saves, and historical replay.
`HybridRetriever.retrieve()` already explicitly disables sparse fallback.
This is not a defect and its ranking algorithm will remain intact.

The serving path is FastAPI plus Streamlit. The Dockerfile serves FastAPI;
external BM25 assets and credentials are prerequisites. Public deployment,
traffic behavior and factual answer quality are not established by unit tests.
The repository also describes a future 120-question evaluation; the supplied
30-question set cannot establish that larger release gate.

Older one-shot evaluation and optional reranker/LLM-judge scripts overlap with
the resumable workflow but serve separate experiments. They are not removed.

## Confirmed findings and change plan

Each row records the problem, source evidence, affected behavior, risk, and
planned validation before implementation.

| Problem and evidence | Smallest planned correction | Risk and validation |
| --- | --- | --- |
| `evaluate_answers.capture()` only saves after the combined pipeline returns; no retrieval checkpoint is written | Coordinate the existing resumable retriever and shared replay composer; save retrieval first | Resume after mocked generation 429/503, forbid repeated retrieval, preserve prior captures/forms |
| Resumable retrieval skips questions without expected chunks; replay excludes empty rankings | Retrieve every pending question independently of gold answerability; distinguish actual empty retrieval from old skipped records | Test unanswerable/no-gold-chunk cases, true empty results, legacy compatibility; no gold labels in production |
| Composer raises on empty evidence | Return an explicit evidence-insufficiency response without calling generation | Test live and replay empty evidence; no citations or invented facts |
| Retrieval summary prints unconditional `PASS` | Report processing completion and measured ranking metrics separately | Test summary wording; do not invent quality thresholds |
| Retrieval CLI defaults to unlimited queries; embedding SDK defaults to five attempts | Bound new questions and provider attempts; stop on first quota failure | Inspect installed SDK implementation and test constructor policies and resume behavior |
| Historical checkpoints have no model/build provenance | Add optional metadata to new records while preserving legacy records exactly | Test legacy/new loads, corpus mismatch, stale/corrupt/duplicate records |
| Citation builder ignores malformed/unbracketed evidence references | Reject invalid references before returning an answer | Test malformed and unknown IDs alongside valid grouped citations |
| No unified question-level progress report | Add offline status and evidence-review materials without assigning grades | Test missing/retrieved/captured/reviewed distinctions; preserve review forms |

## API-consuming paths

- `GeminiEmbeddingClient.embed_documents()` and uncached `embed_query()` call
  Gemini. Dense indexing is explicitly out of scope unless independently needed.
- `GroundedAnswerGenerator.generate()` calls Gemini generation.
- The optional answer judge invokes a separate generation chain; it will not
  be used to manufacture human scores.
- Qdrant construction/queries/scrolls may make remote requests. BM25, corpus
  checks, checkpoint replay construction and local tests require no model calls.
- Initial audit and baseline tests used **zero external API requests**.

The optional cross-encoder reranker may download model assets when explicitly
initialized. It is not part of the serving pipeline and was not initialized.

## Execution order

1. Add focused regressions and correct the confirmed workflow defects.
2. Run all offline tests; validate corpus, backup, references and saved answers.
3. Try Q006 once: persist strict retrieval, then generate from its checkpoint.
4. If successful, proceed in bounded batches and stop at the first quota or
   provider failure; preserve all successful work.
5. Produce a per-question report and evidence review packet. Human quality
   judgments remain unassigned until a real reviewer supplies them.
6. Record final changes, tests, API usage, preservation and unresolved gates.

## Implemented changes

- `scripts/evaluate_answers.py`: live capture coordinates existing resumable
  retrieval and replay, validates corpus before spending requests, and exposes
  offline status and review-packet output. New chunks/checkpoint CLI options now
  apply to live capture as well as explicit replay.
- `scripts/evaluate_hybrid_retrieval_resumable.py`: default one-query budget,
  real retrieval for questions without gold chunks, payload/corpus validation,
  optional provenance on new records, clear corrupt-line errors, and honest
  processing-completion terminology. Existing atomic save mechanism retained.
- `src/evaluation/replay.py`: legacy/new checkpoint compatibility, explicit
  distinction between skipped and truly empty retrieval, provenance corpus check.
- `src/evaluation/progress.py`: question-level state definitions, evidence
  comparison, explicit observed blocker annotation and unscored review packet.
- `src/rag/answer_composer.py`: same evidence-only abstention path for live and
  replay when context is empty; no generation call or fabricated citation.
- `src/generation/citation_builder.py`: reject malformed/unbracketed references
  and duplicate evidence identifiers before returning an answer.
- `src/generation/answer_generator.py`: a raw generation HTTP 429 is reported
  as generation quota failure, not mislabeled as an embedding error by the API.
- `src/indexing/embedding_client.py`: one SDK attempt, with a bounded timeout;
  no model or credential changes. The pinned LangChain generation implementation
  already maps `max_retries=1` to one total SDK attempt.
- `.dockerignore`: retained existing exclusions and added local environments,
  backup/evaluation files and `.env.*`; compatible BM25 assets remain included.
- Focused tests cover failure/resume, provider failures, preserved answers and
  review forms, strict behavior, empty evidence, malformed citations, corruption,
  duplicate IDs, provenance mismatch and atomic-write interruption. The entire
  unit suite now refuses network connections.
- README, architecture and evaluation docs now point to the supported workflow.

## Controlled live result and API accounting

On 2026-11-19 at about 00:54 IST, one bounded Q006 attempt was made after
offline validation. A read-only live Qdrant check returned **1,985 points,
768-dimensional vectors, Cosine distance**. This check established collection
configuration/count; it was not a fresh full-vector comparison.

The first query embedding returned **HTTP 429**. No Q006 retrieval completed,
no generation was attempted, and no further model requests were issued.
The specific quota category and reset time were not established; do not assume
it was a daily quota or repeatedly retry based on a guessed reset time.

HTTP request instrumentation recorded:

| External operation | Attempts | Result |
| --- | ---: | --- |
| Gemini query embedding | 1 | HTTP 429 |
| Gemini generation | 0 | Not attempted |
| Qdrant reads | 3 | HTTP 200 |
| Other external service requests | 0 | None |

The local ledger is `data/processed/live_request_log.jsonl`. It stores category,
timestamp and status only, without credentials or request bodies. No model
switch, key rotation, index recreation, embedding rebuild, commit or push was
performed. The live success gate remains **unmet**.

## Offline evidence and answer review

- All 61 gold chunk references resolve; there are 53 unique referenced chunks.
- The existing BM25 payloads and ordering exactly match the 1,985-chunk corpus.
- Existing vector backup: `BACKUP VERIFIED OFFLINE: 1985 points`; restore dry
  run used no Qdrant access, model requests or writes.
- All five saved answers pass mechanical citation checks; selected evidence
  matches the exact corpus. Human reviews remain **0/30**.
- Q002 omits the functional-specialist-authority expected fact even though it
  is present in the selected evidence. See [source comparison](ANSWER_REVIEW_FINDINGS.md).
  Existing answers were deliberately preserved; they were not regenerated.
- Full human review is required for factual correctness, faithfulness,
  completeness, relevance, citation support and abstention. No automatic or
  assistant observation has been written into human review forms.

## Remaining gates and operational limits

1. Embedding quota/access must become available before Q006 can finish the live
   smoke test. Generation availability is still untested in this work session.
2. Complete Q006–Q030 in bounded batches after one successful smoke test.
3. Independently review all 30 answers; resolve the Q002 completeness finding.
4. Verify deployed FastAPI/Streamlit behavior and public hosting before claiming
   a deployed application. No deployment was performed. Docker exclusions were
   inspected, but no container image was built.
5. The wider 120-question plan still needs manually vetted held-out questions.
6. Legacy optional A/B scripts `evaluate_answer_quality_ab.py` and
   `compare_reranker_answer_case.py` reference removed `DEFAULT_RERANK_LIMIT`
   and `pipeline.reranker` members. They are not supported by the current v1
   pipeline and were not run or rewritten. Use the documented capture/review
   commands; optional reranker/LLM-judge experiments are not release evidence.
7. Run only one evaluator process against a checkpoint/cases pair at a time.
   Atomic replacement protects interrupted saves; it is not a multi-writer lock.
8. Historical Q001–Q005 retrieval has unknown complete model/build provenance.
   New records add actual configuration and code/BM25 hashes; old records are
   neither rewritten nor relabeled as freshly evaluated retrieval.

## Exact resume and maintenance commands

From Ubuntu WSL, using the existing environment:

```bash
cd /mnt/s/github/My-Projects/Generative-AI/enterprise-rag-assistant-v1
RAG_PYTHON="$HOME/.venvs/enterprise-rag-v1/bin/python"
RAG_CHUNKS=/mnt/c/Users/ravit/Downloads/rag-v1-retrieval-artifacts/processed/chunks.jsonl

# Offline: no model or Qdrant requests.
"$RAG_PYTHON" -m pytest -q
"$RAG_PYTHON" -m scripts.evaluate_answers status --chunks "$RAG_CHUNKS" \
  --output data/processed/evaluation_progress.json \
  --review-packet data/processed/answer_review_packet.md
"$RAG_PYTHON" -m scripts.evaluate_answers inspect Q002
"$RAG_PYTHON" -m scripts.evaluate_answers report

# Only after quota/access is available: finish one pending question.
"$RAG_PYTHON" -m scripts.evaluate_answers capture --max-new 1 --chunks "$RAG_CHUNKS"

# After that smoke test succeeds, an explicit small batch; stop on first failure.
"$RAG_PYTHON" -m scripts.evaluate_answers capture --max-new 3 --chunks "$RAG_CHUNKS"

# Optional generation-only replay; use only when status shows a saved ranking
# without an answer. This command makes no new query embeddings.
"$RAG_PYTHON" -m scripts.evaluate_answers capture --max-new 1 \
  --replay-hybrid-checkpoint --chunks "$RAG_CHUNKS"

# Retrieval-only option: at most one new query, saved before any generation.
"$RAG_PYTHON" -m scripts.evaluate_hybrid_retrieval_resumable \
  --max-new 1 --chunks "$RAG_CHUNKS"
```

To record a known quota blocker in a status snapshot, pass
`--retrieval-blocker "First Q006 embedding returned HTTP 429; later questions were not attempted"`.
This annotation is an observed run outcome, not a live quota check. Omit it when
generating a new artifact-only status after the blocker has been resolved.

Serve locally in two terminals using the same environment and existing `.env`
and BM25 assets:

```bash
"$RAG_PYTHON" -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
"$RAG_PYTHON" -m streamlit run src/ui/app.py
```

The app serves fictional policies. Local addresses do not establish a public
deployment. Preserve the existing archive and vector-backup manifest. Review
JSON files are user-owned; automatic commands never assign human ratings.

## Final verification

| Check | Observed result |
| --- | --- |
| Baseline tests | 77 passed |
| Focused pre-fix regressions | 6 failed, 10 passed: reproduced empty-evidence/citation failures and missing validation diagnostics |
| Final full offline suite, with network blocked | **94 passed in 5.22 seconds** |
| Dependency compatibility | `python -m pip check`: no broken requirements |
| Compile | `python -m compileall -q src scripts tests`: passed |
| Diff whitespace validation | `git diff --check`: passed; only Git's existing LF/CRLF normalization notice |
| Gold and corpus reference validation | 30 questions, 121 facts, 61 references, all source IDs resolve |
| Sparse index compatibility | All 1,985 payloads and their ordering match the exact corpus |
| Offline vector restore validation | 1,985 points verified; no restore writes |
| Final protected-artifact SHA-256 comparison | **All 33 files unchanged**, including whole answer/checkpoint files, gold and all review forms |
| Captured/retrieved coverage | 5/30 each; Q006 blocked at embedding, Q007–Q030 unattempted |
| Independent human review | 0/30; no scores fabricated |
| Live end-to-end smoke test | **Blocked**, not passed: embedding HTTP 429 |

Every question is accounted for in [the 30-question status table](EVALUATION_PROGRESS.md).
The machine-readable status, 126,789-byte evidence-review packet, original
artifact hash snapshot and request ledger are under ignored `data/processed/`.
Code/doc/test changes remain uncommitted for review; the user's pre-existing
`.gitignore` change and original backup remain in place.

The reliability corrections and offline validation are complete for the changes
listed here. The **project is not complete**: quota-blocked capture, independent
answer-quality review and deployment verification remain outstanding.
