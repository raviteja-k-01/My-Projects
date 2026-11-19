# Offline engineering review — 2026-11-19

This phase made **zero Gemini or remote service requests**. Provider boundaries
were mocked; the unit suite blocks socket connections. The Docker inventory
probe used only the local named pipe.

## Semantic abstention remains unresolved

Q026 and Q028 fixtures contain their actual gold-referenced source payloads.
Q027 uses nonempty security-related evidence that cannot establish an exact
fine. These are test fixtures, not newly retrieved results. Only question text
and selected context reach production generation, never gold labels or facts.

Twelve cases cover three questions, live/replay composition, and compliant or
adversarial mocked generation through the real generation adapter. Compliant
insufficiency responses are preserved without fake citations. Unsupported
deadlines, fines and client requirements with valid Evidence 1 citations also
pass through. This is a **confirmed semantic validation gap**. Passing tests
named `known-semantic-gap` characterize the weakness; they are not evidence of
safe abstention or Gemini behavior.

General production instructions now distinguish related evidence from support
for requested details, prohibit inferring exact/entity-specific facts, and ask
for relevant responsibilities and conditions. Real-model effectiveness is
untested. No lexical guesswork, gold shortcut or unvalidated judge was added.
Semantic abstention is **not solved**.

## Existing answer review

All five answers and selected evidence remain unchanged. Q001 includes its
three expected elements; Q003 includes both obligations; Q004 includes the six
password/MFA rules; Q005 includes its four core expense requirements. Q005's
additional cited material still needs human relevance/scope review.

Q002's functional-specialist authority fact is explicitly present in selected
Evidence 1 and Evidence 2. Its saved answer includes only commander and
executive-group activation bullets. The omission is confirmed and cannot be
attributed to missing retrieval or discarded context. Its severity relative to
the question's wording remains a human-review decision. No scores were assigned.
See [the source comparison](ANSWER_REVIEW_FINDINGS.md).

## Diff review and corrections

Reviewed production/evaluation changes, tests, new progress reporting, Docker
exclusions and documentation, preserving the user's `.gitignore` edit.

- Ranking/context algorithms and shared composition remain intact. Replay
  constructs context twice to avoid initializing generation on empty evidence;
  this small deterministic duplication does not justify an interface rewrite.
- Corpus/payload, stale-gold and duplicate-ID checks remain enforced. Optional
  new provenance does not rewrite historical records or attest remotely to model
  identity. Atomic replacement remains single-writer, not a concurrency lock.
- Query SDK calls use one attempt. The separate document-indexing outer retry
  policy is unchanged and was not exercised. No model/credential/index changes.
- Citation checks validate references, not entailment; adversarial tests expose
  that limit explicitly.
- Fixed report destination protection: status output cannot replace gold,
  captures, checkpoint, corpus or review files, and two reports cannot share a
  destination. Rejection occurs before preparation or writes.
- Fixed FastAPI lifespan cleanup on exceptions and cleared the global pipeline
  after shutdown. Changes remain uncommitted.

## Offline deployment evidence

Tests exercise real pipeline construction with only provider boundaries mocked,
FastAPI health, `/ask`, normal and exceptional cleanup. Streamlit AppTest checks
initial rendering, submission, success and HTTP 503 presentation with mocked
HTTP. Initial AppTest relative-path failures were corrected to an absolute path.

Existing BM25 loads and searches locally; all 1,985 payloads match the verified
corpus. Docker context retains required BM25 assets and excludes secrets,
environments and evaluation backups. Dockerfile starts FastAPI only; Streamlit
runs separately. Runtime requires GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY and
compatible BM25 artifacts. Secrets must be supplied at runtime. No credential
values were printed or changed.

Docker CLI exists, but its Linux engine named pipe is absent. No image build,
dependency download, container execution or public deployment was performed.
An installed WSL environment does not establish a clean container build.

## Legacy experiments

The two scripts require removed reranker interfaces and optional cross-encoder/
LLM-judge functionality outside the supported v1 workflow. Original bytes are
preserved in `scripts/legacy/*.py.txt`; former command paths now exit 2 with a
clear notice and supported command route before importing any model clients.
No reranker was implemented.

## Preservation and remaining gates

All 33 protected hashes match the original baseline. Five historical checkpoints
remain readable against the verified corpus. Coverage stays 5/30 retrieved,
5/30 captured, 0/30 independently human-reviewed. Prior progress snapshots and
review packets were not overwritten in this phase.

Outstanding: real-model semantic-abstention validation, remaining live captures
when quota and authorization permit, independent answer review, container tests,
and actual deployment verification. The project is not complete.

## Final validation

- Full offline suite with network blocked: **117 passed in 6.90 seconds**.
- `pip check`: no broken requirements; source/scripts/tests compilation passed.
- `git diff --check`: passed (only Git newline-normalization notices).
- Final post-test SHA-256 check: all 33 protected files unchanged.
- Both archived legacy sources match their original Git source content.
- No model calls, remote service calls, dependency installs, commits or pushes.

The 117 passing tests include intentional characterization of the unresolved
semantic gap; this count is not an answer-quality or abstention-accuracy score.
