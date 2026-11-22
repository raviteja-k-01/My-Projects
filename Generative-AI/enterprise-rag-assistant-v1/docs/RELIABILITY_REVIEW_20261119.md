# Reliability review — 2026-11-19

**2026-11-22 reconciliation:** The verified Groq development checkpoint
contains 80/80 answers. The 2/80 result refers only to the separate legacy
Gemini checkpoint. The Groq checkpoint is not a human-reviewed quality result:
its review forms remain uncompleted and all quality metrics remain null.

This continuation starts at commit `944e41e` with a clean working tree.
The freshly executed offline baseline is **166 passed (21.32 s)** in the
existing WSL Python environment. Historical results are not new measurements.

## Confirmed defects and validation plan (before implementation)

| Finding | Expected behavior | Validation |
| --- | --- | --- |
| Generation handles 503/504 but has no configurable backoff or shared request budget | Explicit total-attempt cap, single SDK attempt, bounded backoff for transient errors; stop on 429/auth; selected credential never rotates | Inject transient/timeout/quota failures; count exact calls and sleeps; confirm SDK policy from installed source |
| Replay isolates model names only, allowing prompt/config changes in one experiment | Lock model, generation parameters, prompt, context/composer/citation implementation and input hashes; preserve legacy captures read-only | Resume identical config; reject drift before invocation; ensure legacy bytes unchanged |
| Empty-context abstention returns model=null, rejected by replay | Save and reload deterministic abstention without a provider request | Empty-ranking integration test and resumed report |
| Context word-set deduplication can discard a contradictory amount/negation; only eight candidates are considered before skipping duplicates | Preserve differing policy wording; use lower-ranked candidates to fill the existing eight-evidence cap | Synthetic conflicting policies and rank-nine evidence; offline dev selection comparison |
| BM25 returns zero-score results even for absent vocabulary | No lexical evidence is reported for a query with no term matches | Real local index and small-index tests; compare saved dev rankings (unchanged) |
| API sends arbitrary exception text to clients and labels response validation failures as bad user requests | Sanitized upstream failures; malformed generated output is a server-side error | Failure-injection API tests with secret-like exception text |
| Replay reports no failure categories or request-level latency/error accounting | Persist safe attempt records, including failures, and distinguish request budget from answer count | Fail/retry/resume and budget exhaustion tests |

No benchmark labels, source documents, old checkpoints, embedding format/model,
or vector dimensions will be edited. The held-out questions are not inspected
for tuning. New offline artifacts go under `data/processed/engineering_20261119/`.

## Verified architecture and preserved inputs

- The exact corpus SHA-256 is
  `41b7f8ff1d5db169a7cf5c945dbcc5d655578b6e25bea665c10dc7888a83abf4`.
  It contains 1,985 chunks: 1,766 body and 219 table chunks from 12 PDFs.
- All 1,985 BM25 payloads and their ordering exactly match that corpus.
  The existing tokenizer keeps hyphenated enterprise terms; it was not changed.
- The saved vector backup passes integrity/corpus validation for 1,985 points,
  `gemini-embedding-2`, 768 dimensions. No live Qdrant state was re-queried.
- RAG-120 dataset hash:
  `040520cd1c25a006216e345cff6001b696838c68522421a40205a205331b9086`.
  Saved dev retrieval hash:
  `71a3ba874adec552d9792f4ca366b31cddd2f2afe99b4c03464c756caaf44664`.
- 80/80 development retrievals and 2/80 development answers were loaded and
  validated offline. These are historical provider results, not new requests.
  The 40 held-out questions were not used for tuning or live evaluation.
- A SHA-256 comparison confirms **62 protected existing files unchanged**,
  including PDFs, datasets, BM25 artifacts, answer/retrieval checkpoints and
  review forms. The external indexed corpus hash is unchanged too.

## Implemented changes

1. Generation supports explicit authorized key selection and configurable
   model, thinking level, output limit, timeout and total attempts. SDK retries
   remain one attempt; application retries use bounded exponential backoff
   with jitter only for transient failures. Quota/auth failures never rotate
   credentials or retry. Both installed SDK HTTP transports are covered.
2. A shared evaluation request budget counts initial requests and retries.
   Started/completed/failed request events are fsynced to a safe JSONL ledger;
   provider response latency and error categories are separate from answer
   assembly failures. A crashed in-flight request remains visibly unfinished.
3. Experiment fingerprints include actual model/configuration, prompt, relevant
   source code, SDK versions and input hashes. Legacy answers remain readable;
   changed configurations cannot append to them. Single-writer locks, atomic
   saves, source/output separation and provenance validation prevent mixing or
   overwriting completed work. Empty-context abstentions now round-trip.
4. Context deduplication preserves differing wording, including conflicting
   numbers and negations. It can refill from lower-ranked candidates while
   keeping the eight-evidence and estimated body-token caps. BM25 excludes
   zero-score results and handles corpora smaller than the requested K.
   Duplicate entries within one RRF source are rejected, not double-counted.
5. Malformed citations, inconsistent citation text, and changed source metadata
   are rejected. API and evaluator CLI failures do not expose raw provider
   exception chains. Generated response validation errors return a server-side
   error, rather than blaming the user's valid question.
6. Empty, blocked or token-truncated outputs are rejected. The prompt identifies
   evidence as untrusted data; this instruction is not treated as proof against
   prompt injection or hallucination. UI waiting time now accommodates bounded
   backend generation attempts. The default generation timeout remains 120 s:
   one historical successful answer took 64.5 s, so a 60 s default would regress it.
7. Source-locked human annotations now support required/optional facts,
   claim-level support, actual claim-to-citation support, unsupported claims,
   citation coverage and abstention accuracy. Missing reviews yield null quality
   ratios. Required/optional distinctions are proposed separately in
   [development label observations](BENCHMARK_DEV_LABEL_REVIEW_20261119.md).
8. Added a secret-free environment example, explicit generation-key preflight,
   offline dev snapshot/comparison scripts, and updated deployment/evaluation
   documentation. Existing monorepo CI was inspected; no outside-project file
   was edited and no remote workflow was started.

## Measured development comparison

The same 80 saved rankings and exact corpus were used in both configurations.
Source snippet matching uses the existing generic text/table matcher, without
question-specific rules. No model requests or new relevance labels were used.

| Diagnostic | Before | After |
| --- | ---: | ---: |
| Supplied source anchors matched in selected evidence | 73 / 101 | 73 / 101 |
| Mean selected evidence chunks | 5.275 | 8.0 |
| Questions losing a previously matched source anchor | — | 0 |
| Changed evidence selections | — | 77 / 80 |

**No retrieval or answer-accuracy gain is claimed.** The safety tradeoff is
more context and potentially higher input-token cost, in exchange for retaining
different policy wording and filling skipped evidence. Real-model completeness,
latency and distraction effects still need measurement. Recall@K/MRR remain
unreported because independently verified RAG-120 relevance labels are absent.
The evaluation code calculates them once those labels are supplied.

## Verification evidence

- Baseline: **166 passed in 21.32 s**.
- Expanded WSL suite: **211 passed in 25.98 s**, followed by successful source,
  script and test compilation. All suite network connections are blocked.
- Updated API image: **63 passed in 12.88 s** with `docker run --network none`.
  These exercise real API/composer/context/citation/UI code with provider
  boundaries replaced by fixtures; they are not real Gemini quality tests.
- Updated UI image: startup, form submission, answer and citation rendering
  passed under Streamlit AppTest with network disabled. No visual browser QA
  or real provider submission is claimed.
- `docker compose config --quiet` and both Compose image builds succeeded.
  Existing local API and UI health endpoints returned `ok`; those existing
  services were not restarted into the new images.
- Local preflight: 1,985 BM25 chunks; required environment names present;
  network requests zero. `pip check`: no broken requirements.
- Vector-backup restore dry run: 1,985 points verified; no Qdrant/Gemini access
  and no restore writes. Repository whitespace check passes under its normal
  Windows line-ending configuration.
- After correcting preflight to accept an explicitly selected secondary key
  without requiring an unused primary key, **16 preflight tests passed in
  1.14 s in WSL and 0.51 s in the rebuilt API image**. This includes two new
  credential-selection cases. No generation or retrieval code changed after
  the 211-test full-suite run or the 63-test container integration run.

Local evidence files (generated, gitignored):
`data/processed/engineering_20261119/baseline.json`, `evidence_comparison.json`,
`offline_verification.json`, `historical_answer_reviews.jsonl`, and
`docker_packages.txt`. The package snapshot records the resolved Docker build;
direct dependencies are pinned, but the base tag/transitive resolver are not a
fully immutable cross-platform dependency lock. No remote CI run is claimed.

## Gemini failures and external gates

Installed LangChain code maps `max_retries=1` to
`HttpRetryOptions(attempts=1)` and converts the timeout from seconds to
milliseconds. SDK default retries therefore cannot multiply the explicit
application budget. Transport and provider errors have separate categories.
Google recommends bounded backoff and jitter for transient failures; the
project deliberately stops on 429 instead of guessing whether a limit is
temporary or daily. [Provider troubleshooting](https://ai.google.dev/gemini-api/docs/troubleshooting).

Quota limits apply to projects rather than individual API keys; explicit
credential selection does not create extra quota for the same project.
No automatic key failover is implemented or implied.
[Provider rate-limit documentation](https://ai.google.dev/gemini-api/docs/rate-limits).

No Gemini/embedding/Qdrant requests were made during the original offline work.
The user subsequently authorized one development smoke test, at most three
generation attempts, existing `GEMINI_API_KEY`, `gemini-3.6-flash`, and maximum
1,024 output tokens per attempt, reusing saved retrieval in a separate experiment.
The completed capture was found on disk when that authorization was handled;
it was verified offline instead of issuing a duplicate paid request. Results
are recorded below. Historical 503/504 errors do not establish an exact root cause.

## Authorized smoke-test result

The existing capture's ledger records E001 on **2026-11-19, 21:31:22–21:31:45
IST** (16:01:22–16:01:45 UTC). It contains one started request, one successful
response and one saved answer, with no failed or unfinished attempts.

| Item | Verified result |
| --- | --- |
| Recorded provider requests | **1 of 3 authorized**, no retries needed |
| Generation model / credential selector | `gemini-3.6-flash` / `GEMINI_API_KEY` |
| Limits in saved configuration | 3 attempts; 120 s timeout; 1,024 output tokens; SDK attempts 1 |
| Provider response time | 22.892 s |
| Replay assembly + generation | 22.921 s; excludes historical retrieval |
| Reported usage | 2,802 input tokens; 437 output tokens; 3,239 total |
| Selected evidence / mapped citations | 8 evidence chunks / 4 citations |
| Retrieval | Saved strict-hybrid rankings reused; no embedding or Qdrant call |
| New requests during verification | **0**, enforced by a network connection blocker |
| Preserved original files | All 62 protected files unchanged |

The current prompt/configuration, generation SDK versions and relevant source
hashes exactly match the saved experiment. Reconstructing the selected evidence
from the locked corpus and ranking produces exactly the recorded evidence.
The report loader accepts the answer's citation syntax, IDs and source metadata.

Experiment fingerprint:
`ee74b40eca800b417d2b9a1171a10cd386b39bde4dda6239cf5b82849197564f`.
Answer-file SHA-256:
`193a16d7543549a95d47d64da28ef31d3f06c512f34154632c625044469fb693`.
Files under `data/processed/engineering_20261119/`:
`smoke_gemini_3_6_flash.jsonl`, its `.attempts.jsonl` ledger,
`smoke_verification.json`, and the ungraded `smoke_answer_review.jsonl`.

Assistant source comparison: the answer states PUBLIC information, low-risk
synthetic/test content and the explicit approved-exception condition; all three
appear in cited AI governance excerpts. The added restrictions on credentials,
client data, CONFIDENTIAL and RESTRICTED information appear in cited IAM
evidence. This is an assistant observation, not a completed human review or
an accuracy score. Exception precedence remains subject to the previously
recorded source-interpretation limitation.

This case used 2,802 input tokens versus 1,941 in the historical E001 answer
(**44.36% more**), with eight selected chunks versus five. One sample cannot
establish improved accuracy, a general latency improvement, or sustained
availability. No transient failure occurred, so live retry recovery is still
untested. The new configuration has **1/80** development answers in its own
experiment; the historical experiment still has **2/80**. E001 overlaps both,
so these are not three distinct answered development questions. No remaining
budget was spent on another question.

## Unmet completion criteria

This project is **not declared complete or production-ready**. Remaining gates:

1. Authorize broader development evaluation separately; measure sustained
   availability, actual retry recovery, output quality and latency. One
   successful generation smoke test is now recorded and verified.
2. Complete and independently review development answers and relevance labels.
   Two historical answers are not a statistically meaningful quality benchmark.
3. Validate real-model grounding, conflicts, prompt-injection resistance and
   abstention on nonempty but insufficient evidence. A fabricated claim with
   a syntactically valid citation can still pass mechanical checks; existing
   tests explicitly preserve this known limitation.
4. Freeze the accepted system, then run the untouched held-out 40-question set.
5. Verify a real end-to-end deployed question. Local process health and offline
   fixture tests do not establish a working live provider or public deployment.

No commits, pushes, publishing, reindexing, credential changes or history edits
were performed. All source changes stay inside this RAG project.

## Reproduction commands

From the existing WSL environment in the project directory:

```bash
RAG_PYTHON="$HOME/.venvs/enterprise-rag-v1/bin/python"
RAG_CHUNKS=/mnt/c/Users/ravit/Downloads/rag-v1-retrieval-artifacts/processed/chunks.jsonl
"$RAG_PYTHON" -m pytest -q
"$RAG_PYTHON" -m scripts.deployment_preflight --expect-chunks 1985
"$RAG_PYTHON" -m scripts.evaluate_benchmark120_answers report --chunks "$RAG_CHUNKS"

# New output path required; compare against the preserved before-change snapshot.
"$RAG_PYTHON" -m scripts.compare_dev_evidence --chunks "$RAG_CHUNKS" \
  --baseline data/processed/engineering_20261119/baseline.json \
  --output data/processed/engineering_20261119/evidence_comparison_repeat.json
```

For future changes, `scripts.snapshot_dev_evidence` writes a new source-locked
selection baseline with `--chunks`, `--retrieval`, and `--output`. Both snapshot
and comparison refuse to overwrite an existing output. They are dev-only.

From PowerShell in this project, the API-image integration check can be repeated
without credentials or external network access:

```powershell
$ragProject = (Get-Location).Path
docker run --rm --network none `
  --mount "type=bind,source=$ragProject\tests,target=/app/tests,readonly" `
  --mount "type=bind,source=$ragProject\scripts,target=/app/scripts,readonly" `
  --entrypoint python enterprise-rag-assistant-v1-api -m pytest -q -p no:cacheprovider `
  tests/test_offline_deployment.py tests/test_api.py tests/test_generation_policy.py `
  tests/test_evidence_regressions.py tests/test_context_builder.py `
  tests/test_citation_builder.py tests/test_rag_contract.py tests/test_semantic_abstention.py
```
