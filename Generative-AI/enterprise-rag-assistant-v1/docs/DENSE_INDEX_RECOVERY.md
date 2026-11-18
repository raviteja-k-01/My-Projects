# Recover the full Enterprise RAG dense index (2026-11-18)

This is a **new** Qdrant collection rebuild. The original cloud cluster
`enterprise-rag` was permanently deleted on 2026-10-17 because of inactivity.
Do not assume the old 928-point local SQLite backup is compatible with the
current 1,985-chunk corpus. Do not import it into the new collection.

## Source of truth

- Branch: `rag-v1/reproducible-retrieval-artifacts` (draft PR #1).
- 12 source PDFs, 1,766 body chunks + 219 table chunks = **1,985**.
- `chunks.jsonl` SHA-256:
  `41b7f8ff1d5db169a7cf5c945dbcc5d655578b6e25bea665c10dc7888a83abf4`
- Obtain `processed/chunks.jsonl` from the successful
  `rag-v1-retrieval-artifacts` GitHub Actions artifact on the same branch.
- Required model: `gemini-embedding-2`, 768-dimension document embeddings.
- This is NOT a production readiness certificate: live Qdrant and Gemini
  calls are still required.

## Safety design

- The script refuses missing, partial, or SHA-256-mismatched chunks **before**
  creating any collection or calling Gemini.
- Default mode is offline-only; `--apply` is required for API calls/writes.
- For every existing Qdrant point, the script checks its **integer ID,
  complete metadata/text payload, and stored vector dimension** before reuse.
  It never resumes from the point count alone. Unexpected or changed records
  cause an abort, not an overwrite.
- Upserts are idempotent for the 1-based chunk-file position point IDs.
  A crash after Qdrant accepts a batch can be resumed by reauditing Qdrant.
- The script uses separate `types.Content` objects so Gemini Embedding 2
  generates one embedding per chunk. The returned vector count and all 768
  dimensions are checked before upload.
- Max 16 chunks/call, capped at 2,500 **estimated** tokens/call, with
  12.5 seconds between calls. This is conservative pacing for the AI Studio
  free-tier limits observed on 2026-11-18 (100 RPM, 30k TPM, 1k RPD).
  Estimates are NOT Google token counts. API quotas can still reject calls.
- Per-minute 429 retries are bounded; daily-quota and other errors stop cleanly.
  Saved points remain in Qdrant.
- Final verification rescans **every** Qdrant point (ID, payload, vector
  length), not just total count.

## Operator procedure (Windows PowerShell)

1. Create an **empty new free Qdrant Cloud cluster** in the intended account.
   Configure local-only environment variables `GEMINI_API_KEY`,
   `QDRANT_URL`, and `QDRANT_API_KEY`. Never commit/share keys. Check Qdrant
   cluster availability and Google AI Studio quotas.
2. Use Python 3.11 and the code from the **PR branch**. From this
   `enterprise-rag-assistant-v1` project root, install
   `python -m pip install -r requirements.txt` in a dedicated venv.
3. Run offline preflight:
   `python -m scripts.build_dense_index --chunks "C:\path\to\processed\chunks.jsonl"`
   Expect SHA verification and `DRY RUN`. **No APIs or writes.**
4. Run **one** real smoke-test batch:
   `python -m scripts.build_dense_index --chunks "C:\path\to\processed\chunks.jsonl" --apply --max-batches 1`
   Expect `CHECKPOINT VERIFIED`; independently inspect that the new Qdrant
   collection has valid new vectors.
5. Continue remaining batches in one sitting, if quota permits:
   `python -m scripts.build_dense_index --chunks "C:\path\to\processed\chunks.jsonl" --apply`
   A successful run ends with `PASSED: all 1985 point IDs, payloads and
   vectors verified.`
6. On interruption, rerun the **same** command with the **same** artifact.
   Do not delete the collection or rerun ingestion with changed dependencies.
7. Only after a complete verified rebuild: test live dense retrieval and
   historical hybrid evaluation before deploying the production API.

**Warnings:** Total processing time and free-tier quota use cannot be
guaranteed without a live run; 429 handling does not override daily quotas.
Do not run document ingestion or embedding generation on Uppcoder's limited
free CPU allowance. The 928-point backup is kept intact for historical
reference and is not imported automatically.
