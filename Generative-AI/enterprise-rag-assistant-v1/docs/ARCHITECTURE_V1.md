# Enterprise RAG Assistant V1 — architecture and operational contract

Status: development branch; this file is a design and implementation status
record, **not** proof of production readiness.

## 1. Immutable corpus, offline build, and online serving

```text
OFFLINE / CONTROLLED BUILD (NOT REQUEST PATH)
12 fictional enterprise PDFs (source files in Git)
    -> deterministic extraction / chunking
    -> CI-verified 1,985 chunks; chunks.jsonl SHA-256 locked
    -> offline BM25 index -> local deployable artifact
    -> Gemini Embedding 2 (768 dimensions) -> Qdrant collection
    -> per-point ID/payload/vector audit; safe resumable indexing

ONLINE REQUEST PATH
Streamlit UI (untrusted input)
    -> HTTPS FastAPI /ask
    -> Pydantic input validation + per-client demo rate limit
    -> retrieval:
          same-question embedding LRU cache (up to 512)
          -> Gemini Embedding 2 query vector (768)
          -> Qdrant top-20 dense
          -> BM25 top-20 sparse
          -> RRF fusion
          -> optional BM25-only fallback on Gemini 429 or 5xx
    -> select <=8 evidence chunks within context budget
    -> Gemini generation from evidence only
    -> citation mapping + retrieval_mode (hybrid or bm25_fallback)
    -> answer + source references, or explicit service-unavailable error
```

**No embeddings are generated for 1,985 documents during a user query.**
Document vectors are generated only in controlled indexing.

## 2. Two separate credentials; one model

- `GEMINI_API_KEY`: default credential for embeddings and answer generation.
- `GEMINI_API_KEY_1`: optional additional Gemini credential (for a
  different project/account if provided by its owner).
- `GEMINI_EMBEDDING_KEY_ENV`: optional selector with accepted values
  `GEMINI_API_KEY` (default) or `GEMINI_API_KEY_1`.
- `GEMINI_EMBEDDING_KEY_ENV` selects the **environment variable name**,
  never a literal API key.
- Both choices use `gemini-embedding-2`, the same document/query formatting
  rules, and fixed 768-dimensional output, matching the existing Qdrant index.
- The generation client currently reads `GEMINI_API_KEY` independently.
- **Automatic free-tier key rotation on exhausted project quotas is not
  implemented.** Google documents that quotas are per project and its Google
  APIs Terms prohibit attempts to circumvent documented limits. Consult
  the provider and move to an appropriate paid tier for legitimate sustained
  traffic. Independent projects may have distinct limits; that fact alone
  is not permission to pool their free quotas.
- Never paste keys into logs, issue comments, screenshots, or Git.

Google documentation:
- https://ai.google.dev/gemini-api/docs/rate-limits
- https://developers.google.com/terms

## 3. Failure contracts

| Fault | Application behavior | Evaluation behavior |
| --- | --- | --- |
| Gemini query embedding 429 or 5xx | Return BM25-only evidence; label `retrieval_mode=bm25_fallback` | Fail; never falsely score BM25 as hybrid |
| Bad credential (401/403) | Fail; no silent fallback | Fail |
| Incompatible Qdrant collection | Fail before writes | Fail |
| Gemini answer generation unavailable | Return explicit service-unavailable status via API | Answer evaluation cannot complete |
| Repeated identical question in same process | Reuse valid cached query embedding (512-entry LRU) | Can reuse cached query vectors in same evaluator process |
| Excess public requests from one peer | HTTP 429 with Retry-After | No external API cost |
| Interrupted indexing | Resume verified missing Qdrant IDs, never reset | Not applicable |
| Interrupted retrieval evaluation | Save every finished question; rerun from checkpoint | Reuse previous checkpointed results |

No fake citations or claims of correctness: the citation builder maps evidence IDs,
but support of every factual claim still requires manual and/or dedicated
answer-quality checks.

## 4. Concurrency, scale and security limitations

- The query LRU and rate limiter are **in-process**. Multiple API replicas do
  not share them; a production version needs Redis/edge rate limits and shared
  cache, with a data-versioned cache namespace.
- The rate limiter identifies the request peer. Trust proxy forwarding
  headers only when the deployment proxy is configured correctly.
- `RAG_RATE_LIMIT_PER_MINUTE` defaults to 12 per peer per 60 seconds.
- `/ask` trims and bounds a question to 1–1,000 characters.
- Do not place secret credentials in the Streamlit browser/frontend.
- The Qdrant collection is accessed server-side with a secret key.
- A free Qdrant cluster can be deleted after inactivity; preserve a
  validated offline backup/snapshot and a recovery runbook.
- BM25 assets are intentionally gitignored. **Deployment must receive the
  exact CI BM25 artifacts**; otherwise the RAG backend cannot start.
- Query-embedding cache does not cache generated answers. Generation quota,
  latency and costs still matter.
- The FastAPI service must be load-tested on the actual hosting free tier;
  hosting capacity has not been validated.

## 5. What has been verified

As of 2026-11-18, operator-provided local logs showed:
- 1,985 Qdrant points verified by ID, full payload and vector dimensions.
- 1,985 BM25 payloads exactly match the Qdrant collection (zero mismatches).
- Live hybrid retrieval returned chunks, including SEC-IR-004.
- 29 retrieval-evaluable gold questions (1 unanswerable).
- BM25 on those 29: MRR 0.5087; Hit@5 0.8276; Recall@20 0.6833.
- Live hybrid evaluation has **5 out of 30 checkpointed**; 29-question
  hybrid metrics are NOT yet measured.
- Previous GitHub CI passed 36 tests; reliability-specific tests added later
  must be checked in a fresh CI run and on local WSL.

## 6. Remaining release gates (do not skip)

1. Green CI + local WSL tests after these changes.
2. Verify online BM25 fallback and explicit key selection on the deployed
   backend without leaking secrets or mislabeling evaluation.
3. Run and independently verify the new `scripts.qdrant_backup` vector export.
   Test dry-run restore before considering the index recoverable.
4. Complete all 29 retrieval-evaluable gold questions with the **strict**
   hybrid evaluator; compare matched questions against BM25.
5. Expand to 120 manually vetted gold questions, covering negative/unanswerable
   and cross-document reasoning, before claiming robust accuracy.
6. Test generated answers for factual support, citation precision and
   insufficiency/abstention.
7. End-to-end FastAPI/Streamlit integration, secrets, cold starts, usage
   controls, user-facing error states, and live hosting viability.

## 7. Operator commands (Windows CMD in project directory)

Pull reviewed code:

```cmd
git pull --ff-only
```

Run offline tests in Ubuntu WSL:

```cmd
wsl -d Ubuntu -- bash -lc "cd /mnt/s/github/My-Projects/Generative-AI/enterprise-rag-assistant-v1 && ~/.venvs/enterprise-rag-v1/bin/python -m pytest -q"
```

Check selected credential **name** without printing key material:

```cmd
wsl -d Ubuntu -- bash -lc "cd /mnt/s/github/My-Projects/Generative-AI/enterprise-rag-assistant-v1 && ~/.venvs/enterprise-rag-v1/bin/python -c 'from src.indexing.embedding_client import GeminiEmbeddingClient; c=GeminiEmbeddingClient(); print(\"EMBEDDING_KEY_ENV:\", c.key_env)'"
```

Do not repeatedly run the live hybrid evaluator while embedding quota is
exhausted. Its checkpoint path is `data/processed/hybrid_retrieval_live.jsonl`
and should not be deleted.


## 8. Backup and recovery (no Gemini calls)

A verified JSONL export plus SHA-256 manifest protects against the previous
Qdrant free-cluster inactivity deletion. These files are ignored by Git.
**Do not publish them or commit them to your repository.**

Windows CMD from the project root, using the exact CI-verified corpus:

```cmd
wsl -d Ubuntu -- bash -lc "cd /mnt/s/github/My-Projects/Generative-AI/enterprise-rag-assistant-v1 && ~/.venvs/enterprise-rag-v1/bin/python -m scripts.qdrant_backup export --chunks /mnt/c/Users/ravit/Downloads/rag-v1-retrieval-artifacts/processed/chunks.jsonl"
```

Expect `BACKUP VERIFIED: 1985 points`. It verifies every point's ID,
complete document payload, 768-vector and matching corpus before writing
`data/qdrant_validation/enterprise_rag_chunks_backup.jsonl` plus
`.manifest.json`. This does not reset the collection.

Validate the backup without touching Qdrant:

```cmd
wsl -d Ubuntu -- bash -lc "cd /mnt/s/github/My-Projects/Generative-AI/enterprise-rag-assistant-v1 && ~/.venvs/enterprise-rag-v1/bin/python -m scripts.qdrant_backup restore --chunks /mnt/c/Users/ravit/Downloads/rag-v1-retrieval-artifacts/processed/chunks.jsonl"
```

Expected: `BACKUP VERIFIED OFFLINE: 1985 points` and `RESTORE DRY RUN`.

Only if a cluster is lost, first configure a replacement Qdrant cluster,
then run the **same** restore command with `--apply`. That operation creates
the required 768-cosine collection, skips verified compatible existing points,
uploads only missing points, and audits all restored payloads and vectors.
It will refuse a mismatched corpus, altered backup, or incompatible existing
Qdrant data. Restore never invokes Gemini.

Store an additional private encrypted copy of the backup and manifest outside
the project drive to survive local disk loss.
