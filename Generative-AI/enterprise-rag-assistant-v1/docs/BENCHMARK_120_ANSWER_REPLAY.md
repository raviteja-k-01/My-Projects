# RAG-120: generation-only replay of captured strict-hybrid retrieval

This addition is **separate from the 30-question gold evaluation**. It reuses
**real saved hybrid retrieval rankings** and the existing production
`ContextBuilder`, `compose_answer`, `GroundedAnswerGenerator` and `CitationBuilder`.
The generator receives only the benchmark question plus selected retrieved
corpus text. It never sees benchmark reference answers or expected facts.

## Guarantee and limits

- `preview` / `report`: offline, never initialize the generation service or
  retriever. Hash-verify the benchmark dataset, corpus and saved strict-hybrid
  rankings; validate question identities and known chunk IDs.
- `capture`: requires both `--max-new N` and `--confirm-generation`. Makes up to
  **N new answer compositions**. `--request-budget M` caps total provider
  attempts including retries (defaults to N); the SDK itself makes one attempt. Calls
  **neither query embeddings nor Qdrant**. Saves each successful answer
  atomically; subsequent runs reuse successfully saved answers.
- Default split is `dev`. Held-out `test` remains inaccessible unless
  `--split test --unlock-test` are given, with its *own* test retrieval checkpoint
  and answer output. Do not use the test split for prompt tuning.
- The answer checkpoint is tied to SHA-256 fingerprints of the exact dataset,
  corpus, retrieval checkpoint, prompt, generation settings, context/composer/
  citation code and generation package versions. Any configuration change
  requires a **new answer output file**. Legacy captures can be inspected but
  cannot be appended with unverifiable current settings. Preserve them.
- A `.lock` file prevents competing writers. After a crash, verify that no
  evaluator is running before manually removing its lock. Atomic answer saves
  preserve completed work; `.attempts.jsonl` records request starts/results
  without prompts or credentials. An unfinished start has an unknown outcome.
- Citation-metadata checks are **not proof of factual correctness**.
  Human review of factual completeness, unsupported claims, citation accuracy
  and appropriate abstention is needed.

## Evaluate the Groq-only live generation configuration

The existing replay CLI now accepts **`--provider groq`** on `capture`. This
uses the production `GroqAnswerGenerator` (including citation-format
normalization) with the same `SYSTEM_PROMPT`, `ContextBuilder`,
`CitationBuilder` and `compose_answer` as the live application. A thin
evaluation adapter records the Groq model, settings, prompt/source hashes,
single-request budget and safe request events. **No Gemini generation, Gemini
query embeddings or Qdrant requests** occur during this saved-retrieval replay.

Use a **new, isolated answer file**. Do not combine prior Gemini answers with
Groq results or modify the held-out test split. `GROQ_API_KEY` must be available
in local, gitignored `.env`.

From PowerShell in this project directory:

```powershell
# OFFLINE — check 80 development rankings and source hashes.
wsl -d Ubuntu -- bash -lc "cd /mnt/s/github/My-Projects/Generative-AI/enterprise-rag-assistant-v1 && ~/.venvs/enterprise-rag-v1/bin/python -m scripts.evaluate_benchmark120_answers preview --chunks /mnt/c/Users/ravit/Downloads/rag-v1-retrieval-artifacts/processed/chunks.jsonl --answers data/processed/benchmark120_answers_dev_groq.jsonl"

# LIVE — capture ONE Groq answer, with cumulative cap of five attempts.
# This invocation makes at most one Groq request, not five.
wsl -d Ubuntu -- bash -lc "cd /mnt/s/github/My-Projects/Generative-AI/enterprise-rag-assistant-v1 && ~/.venvs/enterprise-rag-v1/bin/python -m scripts.evaluate_benchmark120_answers capture --provider groq --chunks /mnt/c/Users/ravit/Downloads/rag-v1-retrieval-artifacts/processed/chunks.jsonl --answers data/processed/benchmark120_answers_dev_groq.jsonl --max-new 1 --request-budget 5 --confirm-generation"

# OFFLINE — report answer count, mechanical citation validity and pending reviews.
wsl -d Ubuntu -- bash -lc "cd /mnt/s/github/My-Projects/Generative-AI/enterprise-rag-assistant-v1 && ~/.venvs/enterprise-rag-v1/bin/python -m scripts.evaluate_benchmark120_answers report --chunks /mnt/c/Users/ravit/Downloads/rag-v1-retrieval-artifacts/processed/chunks.jsonl --answers data/processed/benchmark120_answers_dev_groq.jsonl"
```

Use `report --review-export PATH` with a new path to create **blank human
review forms** containing the actual answer, selected evidence and expected
facts. Only independently reviewed answers can establish accuracy. The replay
follows strict saved hybrid rankings and therefore does not test live query
embedding latency, router accuracy or provider uptime. Groq may still fail
or return invalid citations; errors halt and preserve completed captures.

## PowerShell commands from the project directory

```powershell
# OFFLINE: confirm all 80 dev retrievals available, zero model calls
wsl -d Ubuntu -- bash -lc "cd /mnt/s/github/My-Projects/Generative-AI/enterprise-rag-assistant-v1 && ~/.venvs/enterprise-rag-v1/bin/python -m scripts.evaluate_benchmark120_answers preview --chunks /mnt/c/Users/ravit/Downloads/rag-v1-retrieval-artifacts/processed/chunks.jsonl"

# LIVE: explicitly authorize ONE new generation call; zero new embedding calls
wsl -d Ubuntu -- bash -lc "cd /mnt/s/github/My-Projects/Generative-AI/enterprise-rag-assistant-v1 && ~/.venvs/enterprise-rag-v1/bin/python -m scripts.evaluate_benchmark120_answers capture --chunks /mnt/c/Users/ravit/Downloads/rag-v1-retrieval-artifacts/processed/chunks.jsonl --answers data/processed/benchmark120_answers_dev_reliability.jsonl --max-new 1 --request-budget 1 --confirm-generation"

# OFFLINE: progress and citation metadata. No claim of factual quality.
wsl -d Ubuntu -- bash -lc "cd /mnt/s/github/My-Projects/Generative-AI/enterprise-rag-assistant-v1 && ~/.venvs/enterprise-rag-v1/bin/python -m scripts.evaluate_benchmark120_answers report --chunks /mnt/c/Users/ravit/Downloads/rag-v1-retrieval-artifacts/processed/chunks.jsonl"
```

To export unreviewed evidence-backed answer-review records, add
`--review-export /mnt/c/Users/ravit/Downloads/benchmark120_answer_review.jsonl`
to the `report` command, after at least one answer is captured. The export will
**refuse to overwrite** existing files. No review ratings are assigned by code.

HTTP 429 and authentication failures stop immediately without changing keys.
Transient 500/502/503/504 and transport failures can use bounded backoff when
`GEMINI_GENERATION_MAX_ATTEMPTS` is explicitly increased (1–5). The request
budget applies across all questions and retries. A failed answer stops the batch.
Default behavior remains one provider attempt. Model access/price is not
validated by a successful local configuration check.

For a new experiment, pass a new `--answers` filename. Do not append the two
historical development answers after changing the prompt or evidence selector.
The old 30-question workflow is retained for historical inspection; use the
RAG-120 workflow for configuration-locked experiments and request ledgers.

`report --review-export PATH` now includes a blank, answer-hashed `review`.
After an actual human reviews the complete answer and supplied evidence, use
`report --reviews PATH` to aggregate annotations. Required/optional expected
facts are separate; optional facts require rationale. Copy every factual claim
verbatim, including its citation, annotate support, and supply a verdict for
each actual claim-to-citation pair. Set `claims_exhaustive=true` only after
checking all factual claims. Groundedness, completeness, citation correctness,
unsupported claims and abstention metrics have explicit denominators; absent
reviews produce null quality metrics. Assistant/model reviews must not be
marked human. Provider error rates and latency come only from the attempt ledger.
