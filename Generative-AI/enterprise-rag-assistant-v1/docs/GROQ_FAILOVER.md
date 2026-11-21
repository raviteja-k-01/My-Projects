# Optional Gemini → Groq generation failover (live API only)

The live `/ask` pipeline uses Gemini generation as the **primary** model.
If Gemini generation fails with a classified **503, 504, 500/502, transport failure,
or 429 quota error**, the same question and retrieved evidence packet is sent
**once** to Groq's `openai/gpt-oss-120b`. There is **no second retrieval or query
embedding request**. Other failures (bad authentication, bad request, blocked
or incomplete response, exhausted evaluation budget) do not trigger failover.

Failover is **off by default**. To enable it, put these on separate lines in your
local `.env` (or your deployment's secret/environment configuration):

```dotenv
GROQ_FALLBACK_ENABLED=true
GROQ_API_KEY=<your-own-key>
GROQ_FALLBACK_MODEL=openai/gpt-oss-120b
GROQ_FALLBACK_TIMEOUT_SECONDS=35
GEMINI_GENERATION_TIMEOUT_SECONDS=25
```

The Gemini timeout is optional but makes failover start sooner when Gemini
hangs. It may also cause otherwise-successful long Gemini generations to
time out. `GROQ_FALLBACK_MAX_OUTPUT_TOKENS` defaults to 2048 and accepts
values between 1 and 8192.

After setting the key, rebuild the API image because the repository's
`Dockerfile.api` copies `src/` at build time:

```powershell
docker compose up -d --build api
docker compose logs --tail 30 api
```

One Ask request should return the actual model in `model`; if Groq was used,
`generation_fallback_used` is `true` and the Streamlit UI displays a notice.
The API preserves the original citation builder. Failover is not proof that
every answer is fully supported or accurately cited; evaluate the model
separately on the held-out question set.

## Important behavior

- Gemini **embeddings**, Qdrant and BM25 are unchanged.
- The fallback is enabled only in the live pipeline. Saved hybrid retrieval
  and benchmark replay continue to use their explicit generation setup.
- The fallback **is not automatic** until a valid `GROQ_API_KEY` is configured.
  Enabling it without the key makes the backend fail startup rather than
  claiming an unconfigured fallback exists.
- If Groq also fails, the API reports an error; it never fabricates an answer.
- No model calls run in the offline tests:
  `python -m pytest tests/test_groq_failover.py -q`
- Groq free-tier usage and limits are subject to your account and can change.
- Groq receives the retrieved evidence packet. Do not enable it for
  confidential enterprise data without appropriate data-processing approval.

Groq supported model IDs: https://console.groq.com/docs/models
Groq API contract: https://console.groq.com/docs/api-reference
