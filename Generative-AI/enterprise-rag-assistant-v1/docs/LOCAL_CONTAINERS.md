# Local-only Docker Compose smoke test (not public deployment)

This package adds Docker Compose configuration for the existing enterprise RAG
backend and Streamlit frontend. It does not alter retrieval or answer behavior.
The 12 policy PDFs are **fictional**. The BM25 index is local and must be
mounted separately; Qdrant and Gemini are external services.

## Prerequisites

1. Docker Desktop with Linux containers and WSL integration, or Docker Engine
   with Compose v2. If the project is on Windows drive `S:`, verify Docker has
   access to that drive. The engine must resolve `./data/bm25` as a bind mount.
2. A valid **local and gitignored** `.env` in the project root with:
   `GEMINI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`. If configured, also supply
   `GEMINI_API_KEY_1` and/or `GEMINI_EMBEDDING_KEY_ENV`.
3. The exact compatible `data/bm25` index directory. For the evaluated corpus
   it should contain 1,985 chunks; build artifacts are NOT in Git.
4. The existing Qdrant collection; no indexing, re-embedding or restoration is
   performed by Compose.

## Safe checks (no Gemini or Qdrant network calls)

From PowerShell in the project directory:

```powershell
wsl -d Ubuntu -- bash -lc "cd /mnt/s/github/My-Projects/Generative-AI/enterprise-rag-assistant-v1 && ~/.venvs/enterprise-rag-v1/bin/python -m scripts.deployment_preflight --expect-chunks 1985"
```

This verifies environment-variable presence (without showing values) and
locally loads the BM25 index. For an alternate corpus, omit `--expect-chunks`
or configure your actual corpus count. A successful preflight cannot prove
that Gemini or Qdrant is reachable.

Then verify the Compose file **without starting services**:

```powershell
docker compose config --quiet
```

`docker compose config` can interpolate variables; `--quiet` avoids printing
its expanded configuration, which may contain secrets.

## Local container smoke test

Stop existing local FastAPI / Streamlit processes if they occupy 8000/8501.
Then, from the project root:

```powershell
docker compose up --build -d
# Verify Docker service status and local API health.
docker compose ps
Invoke-RestMethod http://127.0.0.1:8000/health
```

Open `http://localhost:8501` in your browser. The API's health endpoint checks
process startup only. A *real question* calls Gemini and Qdrant (and may incur
usage/cost), so **do not submit one** unless explicitly intending a live test.

To stop containers:

```powershell
docker compose down
```

No named volumes are created by this configuration. The BM25 index is mounted
read-only and is not deleted by `docker compose down`.

## Security and release limitations

- Both published ports bind to **127.0.0.1**; this is local-only.
- API keys enter the **api** container via `.env`. No keys are sent to the
  Streamlit container; `RAG_API_URL` uses Compose's private DNS (`api:8000`).
- The API demo rate limiter and embedding cache are process-local. This
  Compose file uses a single backend replica. Public multi-user hosting needs
  a trusted reverse proxy, HTTPS, persistent security controls and distributed
  rate limiting.
- Keep `.env`, BM25 assets, all JSONL datasets/checkpoints, and source PDFs
  out of the Docker build context and images (see `.dockerignore`).
- Preflight is not full readiness: generation 503/504 can still occur.
- The image has not been built or run by the creator of this bundle because a
  Docker daemon is not available in the authoring environment. Treat the
  container smoke test above as a required release gate, not an achieved one.
