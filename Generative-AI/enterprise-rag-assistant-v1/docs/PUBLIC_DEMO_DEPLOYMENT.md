# Public portfolio demo deployment — FREE TIER, LIMITS APPLY

> [!CAUTION]
> **NOT THE CHOSEN PLAN — historical Render + Streamlit proposal.** Do not follow the deployment instructions on this page.
>
> **Agreed target (November 21, 2026):** gated, free-tier, always-on **Cloudflare** deployment. One Worker serves UI and API; Vectorize reuses the existing 768-dimensional Gemini vectors without re-embedding; D1 FTS5 handles keyword search; Gemini provides query embeddings; Groq generates answers with fallbacks; AI Gateway and Turnstile protect the service. Use FastAPI only if it meets the **10 ms Worker CPU gate**; otherwise use a TypeScript Worker. On embedding failure, fall back to D1 keyword-only search. On generation failure, show sources explicitly labeled **unverified**. Preserve the current application as the baseline.
>
> **Status:** Gate 0 complete; Gate 1 in progress. Vectorize Free availability was reported by a Cloudflare dashboard assistant but **has not been verified by provisioning**. No Cloudflare resources have been created. Free tiers have limits and cannot guarantee uninterrupted service.
>
> **This page is retained as an unselected alternative, not a deployment checklist.**


**Not an unlimited or production-ready service.** A free public URL can be
provided for a personal demo, but we cannot guarantee uptime, latency, or
unlimited Gemini/Groq requests. Neither Render nor Streamlit Community Cloud
is an unlimited free production SLA.

## Architecture

- **Streamlit Community Cloud**: browser UI; server-side requests to FastAPI.
  Uses only `streamlit`, `requests`, `python-dotenv` from
  `src/ui/requirements.txt`. No provider credentials or vector store in UI.
- **Render Free web service**: one FastAPI worker on 512 MB RAM. It will sleep
  after 15 minutes idle, then cold start. Full RAM/build compatibility is **not
  established until a real deployment boot succeeds**.
- **Existing Qdrant Cloud cluster**: exact pre-built 768d Gemini-embedding-2
  collection `enterprise_rag_chunks` with 1,985 indexed points. Do not
  regenerate embeddings or overwrite existing vectors.
- **Original BM25S index**: bundled as a separately approved ZIP Release
  asset, SHA-256-checked before load on every ephemeral Render start.
  **Do not put .env, API keys, user data, backups, or generated answers in ZIP.**
- **Gemini API**: live query embeddings; free API quotas still apply.
- **Groq**: `openai/gpt-oss-120b` generation; free API quotas still apply.

## Non-negotiable limitations

1. Render Free: 512 MB RAM, 0.1 vCPU, 15-minute idle spin-down,
   750 instance-hours/workspace/month and ephemeral filesystem.
   The original backend image was **not** sized on Render's free instance;
   startup may fail due to RAM pressure. Never claim public deployment
   success from a local container test.
2. Groq free `openai/gpt-oss-120b`: documented account-level
   30 RPM / 1000 RPD / 8000 TPM / 200,000 TPD limits (check current
   organization dashboard before usage). A **single** request can also fail.
3. Gemini embedding quota may independently be exhausted. Application
   uses BM25 fallback for embedding 429/5xx, but it is not equivalent quality.
4. Qdrant Cloud free tier: 1 GB RAM / 4 GB disk, single node. Its free
   cluster is suspended after **1 week of inactivity** and deleted after
   **4 weeks of inactivity** if not reactivated. The existing collection's
   actual endpoint/count must be checked; preserve a verified backup.
5. Public demo endpoints have **server-to-server token authentication** and
   process-local rolling global hourly/daily limits. These **reset after
   free-host restart and are not a strict billing or provider-quota guarantee**.
   Do not enable multiple API replicas or expose the secret to browser.
6. Source PDF data is fictional. **You must inspect and approve BM25 contents
   before publishing the archive as a public GitHub release.** If not
   publishable, use a private artifact store instead; do not expose secrets.

## Step A — Package the existing index (local; zero API calls)

From PowerShell at the RAG project root, after `git pull --ff-only origin main`:

```powershell
wsl -d Ubuntu -- bash -lc "cd /mnt/s/github/My-Projects/Generative-AI/enterprise-rag-assistant-v1 && ~/.venvs/enterprise-rag-v1/bin/python -m scripts.deployment_preflight --expect-chunks 1985 && ~/.venvs/enterprise-rag-v1/bin/python -m scripts.package_bm25_for_demo --output data/processed/bm25_demo.zip"
```

The output includes a SHA-256 digest and the gitignored archive path
`data/processed/bm25_demo.zip`. The original `data/bm25` is untouched.
**Never commit the ZIP to Git.** If public policy data is approved, create a
GitHub Release in `raviteja-k-01/My-Projects` and upload the ZIP as a release
asset. Copy its **actual** HTTPS release-asset URL. GitHub release assets
are published at the repository level, not inside the monorepo project folder.
The Release must stay available for every free-host boot.

## Step B — Deploy API to Render

In https://dashboard.render.com/ choose
**New → Web Service → Git repository: `raviteja-k-01/My-Projects`**.

- **Root Directory**: `Generative-AI/enterprise-rag-assistant-v1`
- **Runtime**: Docker
- **Dockerfile Path**: `Dockerfile.public-api` (relative to root directory)
- **Instance**: Free; **1 replica/worker only**
- **Health Check Path**: `/health` (liveness) or `/livez`
- **Readiness Check Path**: `/readyz`
- **Start command**: use Dockerfile default; do not override
- **Environment variables** (set in Render, **never Git**):
  `RAG_GENERATION_PROVIDER=groq`,
  `GROQ_API_KEY=<existing secret>`,
  `GEMINI_API_KEY=<existing embedding secret>`,
  `GEMINI_EMBEDDING_KEY_ENV=GEMINI_API_KEY`,
  `QDRANT_URL=<existing reachable Qdrant Cloud URL>`,
  `QDRANT_API_KEY=<existing secret>`,
  `RAG_BM25_ARCHIVE_URL=<actual HTTPS release asset URL>`,
  `RAG_BM25_ARCHIVE_SHA256=<exact packaging output sha256>`,
  `RAG_DEMO_MODE=1`,
  `RAG_DEMO_API_TOKEN=<unique random secret with 24+ characters>`,
  `RAG_RATE_LIMIT_PER_MINUTE=1`,
  `RAG_DEMO_MAX_REQUESTS_PER_HOUR=6`,
  `RAG_DEMO_MAX_REQUESTS_PER_DAY=20`.

Cloud boot downloads the approved archive and SHA-256 verifies it,
then validates the 1,985 corpus chunks and starts FastAPI. No embeddings
are requested during boot. **If boot fails, inspect Render logs**; do not
disable preflight or upload an unrelated index to make it start.

The resulting URL will have the form
`https://<actual-service-name>.onrender.com` (do not invent the name).
Visiting `/health` must return `{"status":"ok"}`; this only tests process
startup, not real question answering.

## Step C — Deploy UI to Streamlit Community Cloud

In https://share.streamlit.io/,
create an app from the same GitHub repository, branch `main`.

- **Entry point**: `Generative-AI/enterprise-rag-assistant-v1/src/ui/app.py`
- Community Cloud should use the adjacent
  `Generative-AI/enterprise-rag-assistant-v1/src/ui/requirements.txt`,
  not the backend's full `requirements.txt`.
- Add two root-level secrets via **Advanced settings → Secrets**:
  ```toml
  RAG_API_URL = "https://<real-render-service>.onrender.com"
  RAG_DEMO_API_TOKEN = "<EXACT SAME SECRET as Render>"
  ```
  Do not put Gemini, Groq, or Qdrant credentials on Streamlit.
  `RAG_DEMO_API_TOKEN` is only used by Streamlit's Python server in its
  outgoing request header; no browser frontend needs it.

The Streamlit URL is the demo link. Both Render and Streamlit can sleep.

## Step D — Real verification, before sharing publicly

1. Check Render `/health`. Inspect build/boot logs for
   `BM25_ARCHIVE_INSTALLED`, `DEPLOY_PREFLIGHT_OK: BM25 1985`.
2. In the hosted Streamlit page, run **one authorized real policy question**.
   Confirm an answer and source citations in the interface. This can consume
   Gemini embedding and Groq generation quota; do not run an uncontrolled load
   test.
3. Submit a greeting: should use local query router and no model quota.
4. Confirm an unauthorized direct `POST /ask` returns HTTP 401. Do not paste
   the shared token into a public browser console or chat.
5. Check that a second independent user cannot exceed configured limits.
   If API quota is exhausted, the app may return graceful HTTP 503; this is
   expected protection, **not unlimited reliability**.
6. Only then put the `*.streamlit.app` URL on the resume. Describe it as a
   **public portfolio demo**, not a production SaaS deployment.

## Source references (checked 2026-11-22)

- Render: https://render.com/docs/free
- Streamlit Community Cloud: https://docs.streamlit.io/deploy/streamlit-community-cloud
- Groq free rate limits: https://console.groq.com/docs/rate-limits
- Qdrant Cloud free tier: https://qdrant.tech/pricing/
- Gemini limits: https://ai.google.dev/gemini-api/docs/rate-limits

Public provider limits can change. **This project cannot guarantee zero
rate-limit issues on free plans.**
