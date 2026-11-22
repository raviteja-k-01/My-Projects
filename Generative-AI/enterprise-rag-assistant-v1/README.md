# Enterprise Workplace Policy Assistant (RAG)

**[Live demo](https://enterprise-rag-assistant-v1.pages.dev/)** · **[Source code](https://github.com/raviteja-k-01/My-Projects/tree/main/Generative-AI/enterprise-rag-assistant-v1)**

A retrieval-augmented question-answering assistant for **12 fictional workplace policy PDFs**. It searches policy text and tables, generates answers grounded in retrieved passages, and displays source citations. The project demonstrates document ingestion, hybrid retrieval, answer validation, evaluation workflows, and serverless deployment.

> **Portfolio project:** The policies and organization are fictional. Do not use its answers for real HR, security, legal, or compliance decisions.

## At a glance

| Component | Implementation |
| --- | --- |
| Corpus | 12 synthetic PDFs; **1,985 chunks** (1,766 text/body + 219 table) |
| Embeddings | Gemini `gemini-embedding-2`, 768-dimensional vectors |
| Retrieval | Semantic vector search + keyword/full-text search |
| Answer generation | Groq `openai/gpt-oss-120b` in the public demo |
| Public frontend / API | Cloudflare Pages + Workers |
| Public retrieval storage | Cloudflare Vectorize (dense) + D1 FTS5 (keyword and source lookup) |
| Access protection | Cloudflare Turnstile and bounded question sessions |
| Original Python implementation | FastAPI, Streamlit, Qdrant, BM25S, reciprocal rank fusion (RRF) |
| Local validation | Pytest and Docker Compose |

## Live application architecture

```text
Browser (Cloudflare Pages)
          |
          v
   Cloudflare Worker
          |
          +--> Gemini query embedding
          |          |
          |          v
          |     Vectorize search -----+
          |                           |
          +--> D1 FTS5 search --------+--> Retrieved policy evidence
          |                                        |
          +--> D1 source lookup                    v
          |                                 Groq answer generation
          |                                        |
          +<--------- Answer + source citations ---+
```

The deployed assistant uses **hybrid retrieval**: vector similarity finds semantically related passages, while D1 full-text search finds keyword matches. Retrieved policy passages are supplied as evidence to the answer generator. The interface shows source-linked citations rather than a raw chunk dump.

For insufficient evidence, invalid citations, or upstream model failures, the application uses constrained fallback/error responses instead of inventing source references. Citation-reference validation checks that citations point to supplied evidence; **it does not prove that every statement is factually supported**.

### Why both Qdrant and Vectorize?

They belong to **different stages of the same project**, not two vector databases serving one request.

| Stage | Dense retrieval | Keyword retrieval | Application |
| --- | --- | --- | --- |
| Original local development | Qdrant | BM25S | FastAPI + Streamlit |
| Public Cloudflare deployment | Vectorize | D1 FTS5 | Workers + Pages |

The original Python pipeline combines Qdrant and BM25S results with **RRF**. For the serverless deployment, the existing 1,985 Gemini document vectors were migrated to Vectorize; their corresponding text and metadata were loaded into D1. **The public demo does not query Qdrant.**

## Evaluation and observed results

The **RAG-120** evaluation set has **80 development questions** and **40 held-out questions**. The held-out split was kept separate from development analysis.

| Check | Recorded result | What it establishes |
| --- | --- | --- |
| Development retrieval checkpoints | **80 / 80** | Saved rankings exist for all development questions |
| Saved Groq development answers | **80 / 80** | Answer records and mechanical citation-reference validation completed |
| Provisional source-anchor Hit@5 | **53 / 72 = 73.6%** | Source-text anchor diagnostic on answerable development questions; **not independently verified retrieval accuracy** |
| Human factual/faithfulness reviews | **0 completed** | No defensible factual accuracy or faithfulness percentage yet |
| Python automated tests | **345 passed** | Latest recorded local test run (November 22, 2026) |
| Cloudflare Worker tests | **26 passed** | Latest recorded Worker test run |
| Public-demo query latency | **3.25 s** | One observed end-to-end policy-query response; **not an average** |

The saved development answers and retrieval checkpoints were evaluated against the **original Python implementation**. They are not a separate 80-query performance benchmark of the Cloudflare deployment. Mechanical citation validity is not the same as claim-level citation correctness or answer faithfulness.

See [RAG-120 evaluation methodology](docs/BENCHMARK_120.md), [answer replay/review](docs/BENCHMARK_120_ANSWER_REPLAY.md), and the [engineering reliability review](docs/RELIABILITY_REVIEW_20261119.md).

## Repository scope

This directory contains the **original Python RAG application**, ingestion and retrieval code, evaluation scripts, tests, Docker configuration, and engineering documentation:

| Path | Contents |
| --- | --- |
| `src/ingestion/` | PDF parsing and deterministic text/table chunking |
| `src/indexing/` and `src/retrieval/` | Embeddings, Qdrant/BM25S retrieval, fusion |
| `src/rag/` and `src/generation/` | Evidence selection, generation, citation checks |
| `src/api/` and `src/ui/` | Local FastAPI and Streamlit application |
| `src/evaluation/`, `scripts/`, `tests/` | Benchmark utilities, reproducibility checks, tests |
| `compose.yaml`, `Dockerfile.api`, `Dockerfile.ui` | Local containerized setup |

**Source availability:** The live Cloudflare Worker is deployed separately; its runtime source is **not yet committed to this repository**. The local Python code in this directory is not the source used by the public Cloudflare endpoint. The links under `docs/` chiefly describe the local implementation and its historical experiments.

## Run the original Python version locally

**Prerequisites:** Python 3.11+, your own authorized API credentials, the matching **1,985-chunk BM25S artifacts**, and access to the corresponding Qdrant collection. Generated indexes, vector backups, evaluation captures, and credentials are not bundled in Git. A fresh clone alone cannot run the full pipeline.

```bash
git clone https://github.com/raviteja-k-01/My-Projects.git
cd My-Projects/Generative-AI/enterprise-rag-assistant-v1

python -m venv .venv
source .venv/bin/activate    # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env         # Windows PowerShell: Copy-Item .env.example .env
```

In the local, gitignored `.env`, configure `GEMINI_API_KEY`, `QDRANT_URL`, and `QDRANT_API_KEY`. For Groq answer generation, also configure `GROQ_API_KEY` and `RAG_GENERATION_PROVIDER=groq`. See [provider selection and optional failover](docs/GROQ_FAILOVER.md).

After restoring the **matching** BM25S artifacts, run a read-only preflight:

```bash
python -m scripts.deployment_preflight --expect-chunks 1985
```

Then start the two local services in separate terminals:

```bash
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

```bash
python -m streamlit run src/ui/app.py
```

Open `http://localhost:8501`. Local Docker Compose instructions are in [LOCAL_CONTAINERS.md](docs/LOCAL_CONTAINERS.md). **Docker is used for local validation; Cloudflare serves the public demo.**

## Limitations and security

- The corpus is synthetic; this is **not** an operational policy-advice service.
- Retrieved passages can be incomplete or irrelevant, and a syntactically valid citation can still accompany an unsupported claim. Human answer-quality review is unfinished.
- The 73.6% source-anchor diagnostic is **provisional**; no verified Recall@K, MRR, faithfulness score, or production average latency is claimed.
- API quotas and provider availability can affect live responses. Neither a health check nor passing offline tests guarantees live answer quality.
- Never commit `.env` files, API keys, corpus/vector backups, private evaluation checkpoints, or sensitive deployment settings.
