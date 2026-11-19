# Enterprise RAG Assistant V1

**Status:** development/portfolio project; real indexing and hybrid retrieval
have been verified locally. **Not yet validated as a publicly deployed,
end-to-end application.**

An evidence-grounded question-answering assistant for 12 **fictional**
enterprise policy PDFs. It combines dense semantic search, BM25 keyword
search, source-aware answer generation, and a human-reviewed RAG evaluation
workflow. The goal is **correct, relevant, understandable policy answers**
with defensible citations—not a high retrieval metric in isolation.

## Architecture

```text
12 fictional enterprise PDFs
       |
       v
deterministic text/table extraction -> 1,985 identified chunks
       |
       +--> Gemini gemini-embedding-2 (768D) --> Qdrant Cloud
       |
       +--> BM25S local index
                     |
User question --> FastAPI input validation + rate limiter
                     |
       query embedding (512-entry in-process cache)
                     |
                Dense + BM25
                     |
                RRF fusion
                     |
       selected evidence (up to 8 chunks, token budget)
                     |
            Gemini grounded generation
                     |
        [Evidence N] -> document/page/section citation
                     |
          FastAPI result -> Streamlit UI
```

The API labels `retrieval_mode=hybrid` or `bm25_fallback`. If the
embedding API returns a quota/server error, the serving path can degrade
to BM25-only retrieval; evaluation **does not** count that as hybrid retrieval.
Generation can still fail if its own quota is unavailable. API credentials
and Qdrant keys stay server-side.

## Verified build and data

| Item | Verified state |
| --- | --- |
| Fictional source PDFs | 12 |
| Body chunks | 1,766 |
| Table chunks | 219 |
| Total index chunks | **1,985** |
| Qdrant | **1,985** IDs, vectors and payloads verified |
| BM25 | **1,985** chunks, zero Qdrant payload mismatches |
| Recovery | Full-vector JSONL export + SHA-256 manifest and verified offline dry-run restore |
| Tests | GitHub CI and WSL automated tests; check current CI run for exact count |

The backup and original BM25 index are **local/generated assets, not Git
repository contents**. When deploying, provide the exact versioned BM25
artifacts and secret environment variables. The Qdrant recovery script can
restore validated vector/payload backups without re-embedding documents.

## Evaluation: separate evidence retrieval from answer correctness

Existing gold questions: **30**, comprising **27 answerable and 3
unanswerable** questions, with **121 expected-fact checks**. There are 29
retrieval-evaluable questions (Q027 has no expected retrieval chunk).

Current **BM25-only baseline on 29 evaluable questions**:
- MRR: **0.5087**
- Hit@5: **0.8276**
- Recall@20: **0.6833**

This is a **retrieval baseline, not an answer accuracy score**. The current
live strict-hybrid evaluation is only **5/30 checkpointed**, with **5/30
answers captured and 0/30 human-reviewed**. The 2026-11-19 bounded Q006
attempt stopped at the first embedding HTTP 429, before generation.
It is not appropriate to claim a full current hybrid score
until the remaining questions have run. Prior experiment results are not
presented as current verified performance.

The dedicated answer-quality protocol scores:
1. evidence sufficiency;
2. answer relevance;
3. factual grounding;
4. coverage of manually checked expected facts;
5. claim-to-citation support;
6. clear structure;
7. uncertainty handling and appropriate abstention.

Every dimension is graded **0/1/2 by a human reviewer using the actual
evidence supplied to the model**. Automated checks validate citation syntax
and source metadata only; they **do not** certify that the cited text supports
a factual claim. Unreviewed answers never receive fabricated scores.

See [Evaluation protocol](docs/EVALUATION_PROTOCOL.md) and
[Architecture, operational limitations and backup guide](docs/ARCHITECTURE_V1.md).

### Offline answer-evaluation setup

From the project root:

```bash
python -m scripts.evaluate_answers prepare       # 0 API calls
python -m scripts.evaluate_answers inspect Q001   # 0 API calls
python -m scripts.evaluate_answers report         # 0 API calls
```

For all 30 question states and an evidence packet, use the **exact indexed
corpus** (the repository-local chunk file may be absent):

```bash
python -m scripts.evaluate_answers status \
  --chunks /mnt/c/Users/ravit/Downloads/rag-v1-retrieval-artifacts/processed/chunks.jsonl \
  --output data/processed/evaluation_progress.json \
  --review-packet data/processed/answer_review_packet.md
```

To capture **one real end-to-end answer** only when the embedding and
generation APIs have available quota:

```bash
python -m scripts.evaluate_answers capture --max-new 1 \
  --chunks /mnt/c/Users/ravit/Downloads/rag-v1-retrieval-artifacts/processed/chunks.jsonl
```

This command reuses saved retrieval and answers, saves each new strict-hybrid
retrieval **before generation**, then captures through the shared replay path.
After a generation failure, rerunning uses its saved ranking without another
embedding request. Both capture and retrieval-only evaluation default to one
new question. Each model call allows one provider attempt; a quota/service
failure stops the batch. Do not retry a known quota failure blindly.

Questions without expected gold chunks still undergo retrieval; that gold
label never controls production abstention. Actual empty evidence returns a
clear insufficiency response without a generation call. Older skipped empty
checkpoints are distinguished from actual empty retrieval and are not rewritten.

Each successful capture is checkpointed. Review forms are separate JSON files
under `data/processed/answer_review_forms/`; they are never overwritten.
To complete the original 120-question plan, add manually vetted, held-out
gold questions and verify their underlying evidence.

See [engineering audit and resume commands](docs/ENGINEERING_STATUS.md) and
[preliminary source comparison](docs/ANSWER_REVIEW_FINDINGS.md). Source comparison
by an assistant is not an independent human review or an answer-accuracy score.

## Running locally

The project has been tested in Ubuntu WSL with a Linux virtual environment.
Install with a supported Python 3.11+ environment:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

Put credentials in a local **gitignored** `.env`:

```env
GEMINI_API_KEY=your_primary_key
GEMINI_API_KEY_1=your_optional_second_project_key
QDRANT_URL=your_cloud_url
QDRANT_API_KEY=your_cloud_api_key
```

Optional `GEMINI_EMBEDDING_KEY_ENV` chooses the variable name used for
embedding calls (default: `GEMINI_API_KEY`; alternative:
`GEMINI_API_KEY_1`). Both use the **same embedding model** and 768-dimensional
space. This is **explicit selection, not automatic credential rotation**.
Use credentials in accordance with the API provider's usage limits.

The generation model defaults to `gemini-3.6-flash`. Optionally set
`GEMINI_GENERATION_MODEL` in the private `.env` to a different model
**after confirming its availability and quota for the chosen project**.
This does not affect the embedding model or the indexed vectors.
Gemini HTTP 503 means the generation service is temporarily unavailable;
the application reports this separately from embedding HTTP 429.

**Prerequisites:** Your Qdrant collection is indexed and the compatible
`data/bm25/` build artifacts are present. Simply cloning the repository
does not provide these generated resources.

Run FastAPI:

```bash
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Run Streamlit in another terminal:

```bash
streamlit run src/ui/app.py
```

The health endpoint is at `http://127.0.0.1:8000/health`; the app is
normally at `http://127.0.0.1:8501`. These are **local** addresses,
not proof of public hosting.

## Release work still required

The [offline continuation review](docs/OFFLINE_REVIEW.md) records a remaining
semantic gap: unsupported claims with real citation IDs can pass structural
checks. Mocked abstention tests do not establish real-model abstention accuracy.
The retired reranker A/B scripts now redirect operators to the supported
evaluation workflow without initializing services.

- Complete the 29-question strict hybrid retrieval evaluation.
- Capture and **independently review actual generated answers**; investigate
  inaccurate facts, unsupported claims, missing exceptions and false citations.
- Expand to 120 manually verified questions, including holdout, table and
  unanswerable cases.
- Validate generation-model access and response quality on the deployed plan.
- Provision BM25 assets, secrets and cloud hosting; verify cold starts,
  rate limits, error handling and live end-to-end queries.
- Publish a public demo link **only after a real deployment succeeds**.

This is a fictional policy assistant and must not be used as a source of
real corporate, legal, privacy or security policy.
