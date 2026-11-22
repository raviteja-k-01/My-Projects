# Enterprise RAG Assistant

A question-answering app for a library of **12 fictional workplace policy PDFs**. Ask about access, security, HR, or workplace procedures, and the assistant searches the documents before answering with source references.

I built this project to work on a practical RAG problem: finding the right evidence, keeping answers tied to that evidence, and being clear about what the system **hasn't** proved yet. It is a portfolio project, not a real company's policy service.

## What it does

1. **Finds relevant passages.** The documents were processed into 1,985 text and table chunks. Dense search in Qdrant and keyword search with BM25S run together.
2. **Combines the results.** Reciprocal rank fusion (RRF) brings both searches into one ranking. The answer pipeline selects up to eight evidence passages.
3. **Writes an answer.** Groq or Gemini can generate the response using the selected evidence. Gemini embeddings are used for dense retrieval even when Groq generates the answer.
4. **Shows its sources.** FastAPI checks citation references and returns the answer to a Streamlit interface, where the user can inspect the cited evidence.

If there is no usable evidence, the assistant can decline to answer. A valid citation reference is **not** proof that every generated claim is correct; that still needs independent review.

### Main components

| Part | Technology |
| --- | --- |
| Web interface | Streamlit |
| API | FastAPI |
| Dense retrieval | Gemini embeddings (768 dimensions) + Qdrant |
| Keyword retrieval | BM25S |
| Combining results | Reciprocal rank fusion |
| Answer generation | Groq or Gemini, configurable |
| Tests and evaluation | pytest, saved retrieval/answer checkpoints, human-review forms |

## Run it locally

**Requirements:** Python 3.11+, access to the existing Qdrant collection, the matching local BM25 index, and your own authorized API credentials.

Clone the monorepo and enter this project:

```bash
git clone https://github.com/raviteja-k-01/My-Projects.git
cd My-Projects/Generative-AI/enterprise-rag-assistant-v1
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` locally. At minimum, configure `GEMINI_API_KEY` (for query embeddings), `QDRANT_URL`, and `QDRANT_API_KEY`. To generate answers with Groq, also set `RAG_GENERATION_PROVIDER=groq` and `GROQ_API_KEY`. The default generation provider is Gemini.

**A fresh clone is not enough to start the full assistant.** Generated artifacts such as `data/bm25/` and evaluation checkpoints are not included in Git. Use the BM25 artifacts that match the existing 1,985-point Qdrant collection; don't rebuild or replace an index just to get past startup.

Check local configuration and the BM25 index without calling any model or cloud service:

```bash
python -m scripts.deployment_preflight --expect-chunks 1985
```

Start the API:

```bash
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

In a second terminal, with the same Python environment active, start the interface:

```bash
python -m streamlit run src/ui/app.py
```

Open `http://localhost:8501`. You can check API process health at `http://127.0.0.1:8000/health`. A successful health check does **not** verify that Qdrant or a model provider will answer a real question. Live questions use external APIs and may consume quota.

Prefer containers? See the [local Docker Compose guide](docs/LOCAL_CONTAINERS.md).

## Testing and evaluation

The focus is not just whether the app returns an answer, but whether its retrieved evidence and citations can be checked.

- **Corpus:** 12 fictional PDFs, split into **1,766 body chunks** and **219 table chunks** (1,985 total).
- **RAG-120 benchmark:** 80 development questions and 40 held-out questions, kept separate to avoid tuning against the test set.
- **Saved development runs:** 80 retrieval checkpoints and 80 Groq-generated answer records have been mechanically validated. **None of those answers has a completed human factual review.**
- **Automated tests:** **345 passed** in the local WSL environment on **November 22, 2026**.

These results verify code behavior and saved artifact consistency; they do **not** establish a trustworthy answer-accuracy percentage, human-judged grounding score, or live-service uptime. Retrieval relevance and generated claims still need independent review. Earlier 30-question experiments are separate from RAG-120.

For the full methodology and limitations, read the [benchmark guide](docs/BENCHMARK_120.md), [answer replay and review process](docs/BENCHMARK_120_ANSWER_REPLAY.md), and [engineering reliability review](docs/RELIABILITY_REVIEW_20261119.md).

## Deployment status

The planned portfolio setup is a **Render FastAPI backend**, **Streamlit Community Cloud frontend**, and the existing **Qdrant Cloud** collection. The backend needs the approved BM25 archive at startup. Free hosting and model quotas can cause slow starts or failed requests.

**The public demo has not yet been deployed and verified end to end.** A passing local test suite is not a public deployment.

See [public demo deployment instructions](docs/PUBLIC_DEMO_DEPLOYMENT.md) for the actual setup and safety checks. Keep `RAG_ALLOW_EVALUATION_TRACE=false` on a public instance.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/ingestion/` | PDF reading, parsing, and chunking |
| `src/retrieval/` and `src/indexing/` | Search, fusion, and indexes |
| `src/rag/` and `src/generation/` | Evidence selection, answers, and citations |
| `src/api/` and `src/ui/` | FastAPI backend and Streamlit frontend |
| `src/evaluation/` and `scripts/` | Evaluation, validation, and maintenance utilities |
| `tests/` | Automated checks |
| `docs/` | Detailed engineering and deployment notes |

## Boundaries

The PDFs are invented examples, not real workplace policies. The assistant can still miss evidence, misread a condition, or produce a plausible but unsupported statement. Don't rely on it for real HR, security, legal, or compliance decisions.

Never commit `.env`, API keys, private checkpoints, vector backups, or generated evaluation reports. Don't publish an index archive without checking and approving its contents.
