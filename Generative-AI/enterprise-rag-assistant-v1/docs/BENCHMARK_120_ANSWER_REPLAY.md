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
  **N new answer-generation attempts**; Gemini SDK may make retries. Calls
  **neither query embeddings nor Qdrant**. Saves each successful answer
  atomically; subsequent runs reuse successfully saved answers.
- Default split is `dev`. Held-out `test` remains inaccessible unless
  `--split test --unlock-test` are given, with its *own* test retrieval checkpoint
  and answer output. Do not use the test split for prompt tuning.
- The answer checkpoint is tied to SHA-256 fingerprints of the exact dataset,
  corpus and retrieval checkpoint. Generation model changes require a **new
  answer output file**, so different model experiments are not mixed.
- Citation-metadata checks are **not proof of factual correctness**.
  Human review of factual completeness, unsupported claims, citation accuracy
  and appropriate abstention is needed.

## PowerShell commands from the project directory

```powershell
# OFFLINE: confirm all 80 dev retrievals available, zero model calls
wsl -d Ubuntu -- bash -lc "cd /mnt/s/github/My-Projects/Generative-AI/enterprise-rag-assistant-v1 && ~/.venvs/enterprise-rag-v1/bin/python -m scripts.evaluate_benchmark120_answers preview --chunks /mnt/c/Users/ravit/Downloads/rag-v1-retrieval-artifacts/processed/chunks.jsonl"

# LIVE: explicitly authorize ONE new generation call; zero new embedding calls
wsl -d Ubuntu -- bash -lc "cd /mnt/s/github/My-Projects/Generative-AI/enterprise-rag-assistant-v1 && ~/.venvs/enterprise-rag-v1/bin/python -m scripts.evaluate_benchmark120_answers capture --chunks /mnt/c/Users/ravit/Downloads/rag-v1-retrieval-artifacts/processed/chunks.jsonl --max-new 1 --confirm-generation"

# OFFLINE: progress and citation metadata. No claim of factual quality.
wsl -d Ubuntu -- bash -lc "cd /mnt/s/github/My-Projects/Generative-AI/enterprise-rag-assistant-v1 && ~/.venvs/enterprise-rag-v1/bin/python -m scripts.evaluate_benchmark120_answers report --chunks /mnt/c/Users/ravit/Downloads/rag-v1-retrieval-artifacts/processed/chunks.jsonl"
```

To export unreviewed evidence-backed answer-review records, add
`--review-export /mnt/c/Users/ravit/Downloads/benchmark120_answer_review.jsonl`
to the `report` command, after at least one answer is captured. The export will
**refuse to overwrite** existing files. No review ratings are assigned by code.

If generation fails with HTTP 429/503, stop. Existing answer checkpoints are
preserved. The standalone 30-question workflow is unaffected.
