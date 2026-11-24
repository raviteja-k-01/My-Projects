# Enterprise RAG Assistant V1

A retrieval-augmented assistant for 12 **fictional** enterprise policy PDFs.
FastAPI serves retrieval and answer generation; Streamlit presents answers with
source citations. The project demonstrates a complete RAG pipeline, while
independent answer-quality review and public production deployment remain
incomplete.

## Project highlights

- Built a deterministic ingestion pipeline producing **1,985 source-linked
  chunks** from policy text and tables.
- Implemented hybrid retrieval with **Qdrant dense search**, **BM25S**, and
  **reciprocal rank fusion**, followed by bounded evidence selection.
- Connected the retrieval pipeline to **FastAPI** and a **Streamlit** user
  interface, with source references, explicit abstention, and health checks.
- Added reproducible evaluation workflows for the supplied **120-question**
  benchmark, preserving its 80-question development and 40-question held-out
  split.

**Core stack:** Python, FastAPI, Streamlit, Qdrant, BM25S, Gemini Embedding 2,
Groq/Gemini generation, Docker Compose, pytest, and openpyxl.

**Separate experiment:** A Cloudflare Workers version using Vectorize and D1
exists as an alternative implementation; it is not part of the Python
application's request path and has not been production-validated.

## Current evidence

The [engineering report](docs/RELIABILITY_REVIEW_20261119.md) distinguishes fresh
offline checks from historical provider results and remaining release gates.

| Item | Evidence |
| --- | --- |
| Corpus | 12 PDFs; 1,985 source-locked chunks (1,766 body, 219 table) |
| Embeddings | Existing `gemini-embedding-2`, 768 dimensions, unchanged |
| Retrieval | Qdrant dense + BM25S sparse + reciprocal rank fusion |
| RAG-120 | 120 questions: 80 development and 40 held-out; recorded retrieval result: **96.6% Hit@10, 0.64 MRR** |
| Saved development artifacts | 80/80 retrieval checkpoints and 80/80 Groq answer records; these are not 80 live API captures |
| Live API captures | 3 successful responses and 2 recorded failures; the live evaluation is incomplete |
| Human answer review | 0 completed; no semantic accuracy scores assigned |
| Offline tests | 345 passed in Linux Docker on the last verified full run; subsequent rate-limit edits were not fully validated |
| Docker smoke test | API and UI images built; API `/readyz`, `/health`, and UI HTTP 200 verified |
| Answer checks | Citation mechanics passed for 80 saved answers; factual correctness remains unverified |
| Retrieval coverage | 72/72 answerable development questions matched expected document and page; 8/8 unanswerable questions had no expected evidence |
| Evidence comparison | 73/101 source anchors before and after; no per-question regression |
| Docker | Local images build; live provider and public deployment are separate gates |

Anchor matching is a diagnostic, **not Recall@K or answer accuracy**. The recorded
RAG-120 retrieval run reports **96.6% Hit@10 and 0.64 MRR** across the 120-question
dataset. These are retrieval-ranking metrics only; they do not establish answer
correctness, groundedness, citation faithfulness, or abstention quality. The
per-question RAG-120 retrieval output used for the aggregate is not committed,
so the public repository cannot independently recalculate it yet. The separate
30-question gold workflow remains a historical experiment and is not the source
of the 120-question result. The independent relevance-label scorer remains a
separate validation workflow.

## Pipeline and safeguards

PDF text/tables become deterministic chunks indexed in Qdrant and BM25S. Hybrid
results are fused with RRF, then up to eight evidence chunks are selected within
an estimated body-token budget. The configured generation provider (Groq or
Gemini) produces an answer; citation syntax and metadata are checked before
FastAPI returns it to Streamlit.

Keyword-only fallback is explicitly labeled `bm25_fallback`; strict evaluation
never counts it as hybrid retrieval. Zero-score BM25 matches are excluded.
Deduplication preserves differing amounts, conditions and negations. Empty
evidence returns a deterministic abstention. Nonempty-evidence grounding and
abstention still need real-model evaluation and independent review.
For control queries that explicitly name a policy, metadata/table chunks for
that policy are stably prioritized without changing the underlying retrieval
ranking or using evaluation labels.

## Local setup

The working environment is Ubuntu WSL, Python 3.12. Python 3.11+ is supported.

```bash
python -m pip install -r requirements.txt
python -m pip check
python -m pytest -q
```

Copy [.env.example](.env.example) to the gitignored `.env` and fill authorized
credentials locally. Provide the exact compatible `data/bm25/` artifacts and
existing Qdrant collection. A Git clone does not include generated indexes or
checkpoints. Do not reindex just to start the application.

```bash
python -m scripts.deployment_preflight --expect-chunks 1985
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
# In another terminal:
streamlit run src/ui/app.py
```

See [local Docker Compose](docs/LOCAL_CONTAINERS.md) and
[index recovery](docs/DENSE_INDEX_RECOVERY.md). Published ports bind to loopback.
`/health` checks process startup, not provider availability. Real questions use
provider requests. Public hosting would need authentication, HTTPS and suitable
shared controls; the current rate limiter/cache are process-local.
`/livez` is process liveness and `/readyz` verifies that the initialized pipeline
and local BM25 index are ready; `/readyz` is the Docker healthcheck target.

## Generation configuration

| Variable | Default / meaning |
| --- | --- |
| `RAG_GENERATION_PROVIDER` | `gemini` for backward compatibility; set `groq` to use Groq-only answer generation |
| `GROQ_API_KEY` | Required for Groq-only generation or Gemini→Groq fallback; store only in `.env` |
| `GROQ_FALLBACK_MODEL` | `openai/gpt-oss-120b`; model name used in both Groq modes |
| `GROQ_FALLBACK_ENABLED` | `false`; only applicable with `RAG_GENERATION_PROVIDER=gemini` |
| `GEMINI_GENERATION_MODEL` | `gemini-3.6-flash`; confirm project access before use |
| `GEMINI_GENERATION_KEY_ENV` | `GEMINI_API_KEY`; explicit alternative `GEMINI_API_KEY_1` |
| `GEMINI_EMBEDDING_KEY_ENV` | Independent authorized embedding credential selector |
| `GEMINI_GENERATION_MAX_ATTEMPTS` | 1 total attempt; allowed 1–5 |
| `GEMINI_GENERATION_TIMEOUT_SECONDS` | 120 per attempt; allowed 1–120 |
| `GEMINI_GENERATION_MAX_OUTPUT_TOKENS` | 1024; allowed 1–8192 |
| `GEMINI_GENERATION_THINKING_LEVEL` | `low`; blank omits the option for incompatible models |

Gemini generation is never called when `RAG_GENERATION_PROVIDER=groq`.
Gemini query embeddings remain required for dense retrieval. The Gemini SDK
makes one attempt. Optional application retries use bounded exponential
backoff with jitter for transient server/transport failures. Quota/authentication
failures stop immediately. Credentials never rotate automatically. Incomplete,
blocked or empty responses are rejected. Public errors omit raw provider details.

## Live 120-question evaluation automation

The live runner calls FastAPI `/ask` directly, saves each completed response
immediately, and resumes without repeating captured question IDs. It never sends
benchmark reference answers or expected facts to the RAG generation request.
Live calls are preview-only unless `--confirm-live` and `--request-budget` are
both supplied.

```bash
# Offline workbook/status preview: no API, embedding or generation calls
python -m scripts.run_live_evaluation --max-new 0

# Development: explicit authorization and a cumulative request budget
python -m scripts.run_live_evaluation --split dev --max-new 10 \
  --confirm-live --request-budget 10

# Repeat in bounded batches; completed IDs are skipped
python -m scripts.run_live_evaluation --split dev --max-new 10 \
  --confirm-live --request-budget 80

# Freeze development only after 80 actual API captures
python -m scripts.run_live_evaluation --split dev --freeze-dev

# Held-out test is blocked until the development freeze exists
python -m scripts.run_live_evaluation --split test --unlock-test \
  --max-new 10 --confirm-live --request-budget 10
```

Set `RAG_ALLOW_EVALUATION_TRACE=true` only on an authorized evaluation API
instance. Human, AI-judge and citation metrics are kept separate; a completed
human review is required before official quality scores are populated.

## Reproducible development evaluation

From the project root in WSL, use the exact indexed corpus:

```bash
RAG_CHUNKS=/mnt/c/Users/ravit/Downloads/rag-v1-retrieval-artifacts/processed/chunks.jsonl

# Offline historical checkpoint verification:
python -m scripts.evaluate_benchmark120_answers preview --chunks "$RAG_CHUNKS"
python -m scripts.evaluate_benchmark120_answers report --chunks "$RAG_CHUNKS"

# ONLY AFTER spending authorization: separate experiment, at most one provider
# request, zero embeddings and zero Qdrant requests.
python -m scripts.evaluate_benchmark120_answers capture --chunks "$RAG_CHUNKS" \
  --answers data/processed/benchmark120_answers_dev_reliability.jsonl \
  --max-new 1 --request-budget 1 --confirm-generation
```

Experiments lock model/settings, prompt, relevant implementation, SDK versions
and input hashes. Changed or legacy provenance requires a new output filename.
Existing captures stay available for review. Atomic saves and a single-writer
lock protect completed answers. The `.attempts.jsonl` ledger records starts,
results and failures without prompts or secrets. Request budgets include retries.

See [answer replay and human review](docs/BENCHMARK_120_ANSWER_REPLAY.md),
[benchmark workflow](docs/BENCHMARK_120.md), and
[separate label observations](docs/BENCHMARK_DEV_LABEL_REVIEW_20261119.md).
Human annotations distinguish required facts from optional background and assess
claims, citations, unsupported statements and abstention. Missing reviews produce
null quality metrics. Assistant/model judgments must not be labeled human reviews.

## Public portfolio demo (optional, free tiers have limits)

For a split deployment (Render Free FastAPI + Streamlit Community Cloud UI +
existing Qdrant Cloud), see [public demo setup](docs/PUBLIC_DEMO_DEPLOYMENT.md).
It includes the existing BM25 artifact packaging/verification, server-side
shared token, one-instance request caps, and manual live checks. **Not
deployed or load-tested yet.** Free hosting/LLM quotas can still cause cold
starts, request failures or downtime; it is not an unlimited service.

## Remaining release gates

- Complete live development captures when provider capacity and authorization permit.
- Independently review retrieval relevance, answer claims, citations, and abstention.
- Verify real-model abstention and handling of conflicting evidence.
- Freeze a validated configuration before unlocking the 40-question test split.
- Verify a real deployed end-to-end question before claiming a reliable live demo.

The fictional corpus is not guidance about any real company's policies.
