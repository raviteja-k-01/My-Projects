# Enterprise RAG V1 — answer-quality evaluation protocol

**Goal:** Determine whether the assistant produces useful, accurate, grounded
answers with correct citations and appropriate uncertainty. Retrieval rank is
diagnostic evidence, **not** the product's success measure.

Version: 1.0 (2026-11-18). This is a protocol, not a claim that live answers
have already been evaluated.

## Dataset and evaluation layers

The existing gold file contains **30 questions**, **121 expected-fact checks**:
- **27 answerable**, **3 explicitly unanswerable** (Q026, Q027, Q028).
- One of the unanswerable questions (Q027) has no gold retrieval chunk;
  do **not** confuse that with only one unanswerable question.
- Types: factual, semantic, exact keyword, cross-section, cross-document,
  tables, unanswerable, and ambiguous questions.

The evaluation is run in three distinct layers:

1. **Retrieval / evidence availability:** Does strict hybrid retrieval find
   relevant chunks? Record MRR, Hit@5, Recall@20 only as diagnostics.
2. **Evidence sufficiency:** Did the **actual selected 8-or-fewer chunks**
   provide enough information to answer the question or correctly abstain?
   A high ranking score does not guarantee sufficient generation context.
3. **Final answer quality:** Human review of the actual generated answer
   against supplied evidence and the manually checked expected facts.
   Model output is never treated as its own correctness ground truth.

## Human scoring rubric (each dimension 0, 1, or 2)

| Dimension | 0 — Fail | 1 — Partial | 2 — Meets expectation |
|---|---|---|---|
| evidence_sufficiency | Evidence cannot support a correct answer or justified abstention | Evidence covers some required facts but important facts absent | Evidence covers required facts, or justifies why the requested fact is unknown |
| answer_relevance | Off-topic, misunderstands user intent | Addresses part of the question | Directly answers the user's actual question |
| factual_grounding | Material invented/false/contradictory claim | Mostly supported, but some weak or ambiguous claims | Every material factual claim supported by supplied evidence |
| fact_coverage | Omits or contradicts core expected facts | Some expected facts delivered, others missing | All applicable required facts accurately delivered; or correct abstention |
| citation_support | Fabricated citation or material claim cited to wrong source | Some citations valid, material support gaps remain | Every material factual claim cites the exact evidence that supports it |
| structure_clarity | Misleading/unreadable organization | Understandable but unclear, redundant, or poorly ordered | Direct answer, clear ordering, concise and readable |
| uncertainty_calibration | Invents absent specifics or falsely states certainty | Qualifies some uncertainty but over/understates other parts | Appropriately distinguishes known from unknown; abstains when needed |

**Do not calculate a single combined "RAG score".** Report the per-dimension
distribution, fact-level misses and exact failure cases. A model that has
excellent formatting but invents a security policy is unacceptable regardless
of a strong average.

## Expected facts are review prompts, not verified natural-law truth

Each gold row has a list of `expected_facts`, which must themselves be
checked against the appropriate fictional policy documents. Review each fact
with one of:

- `met`: output correctly covers this specific expected fact.
- `partial`: incomplete or qualified coverage.
- `missed`: required fact is absent.
- `contradicted`: generated answer conflicts with the fact/document.

Record any unsupported factual claims in `unsupported_claims` and broken or
misleading claim-to-source attribution in `citation_errors`, with quotes and
reference IDs where possible. For a multi-document question, each relevant
document's contribution must be accounted for; a single citation does not
prove complete coverage.

Unanswerable Q026/Q027/Q028 are **end-to-end answer checks**, not automatic
failures of retrieval: correct behavior is to decline unsupported exact values
without inventing deadlines, fines, or unnamed client requirements. Q026 and
Q028 can legitimately retrieve relevant background chunks and still abstain.

## Workflow: zero API calls until explicit capture

From the project root in Ubuntu WSL:

```bash
python -m scripts.evaluate_answers prepare
python -m scripts.evaluate_answers inspect Q001
python -m scripts.evaluate_answers report
```

`prepare` creates:
- `data/processed/answer_eval_cases.jsonl`: the 30 gold-anchored records.
  Every `run` initially `null` (not yet evaluated).
- `data/processed/answer_review_forms/Q001.json` through `Q030.json`.
  Every rating and expected-fact status initially `null`.

These paths are **gitignored**. The process never erases completed captures
or overwrites review forms. If the gold definition changes for a prior ID,
the evaluation refuses to reuse its stale response.

Only after quota is available and full RAG connectivity is confirmed, execute
an explicit controlled live capture:

```bash
python -m scripts.evaluate_answers capture --max-new 1
```

For each captured question the real pipeline runs **strict hybrid** retrieval
(no BM25 fallback disguised as hybrid), builds its actual evidence context,
generates one answer, maps citations, and atomically saves:
- actual answer and model;
- all retrieved candidate IDs;
- exact evidence text and metadata shown to the generator;
- generated source citations;
- latency/usage if returned;
- independent automatic citation syntax/metadata checks.

A 429 or generation error stops capture. Previously saved cases remain;
rerunning capture skips them. `capture` is deliberately rate-bounded by
`--max-new 1` unless explicitly changed.

For human review:
1. Run `python -m scripts.evaluate_answers inspect Q001` to read the
   **question, expected facts, actual answer, cited sources, and verbatim
   evidence**. Independently check disputed policy facts against PDFs.
2. Edit `data/processed/answer_review_forms/Q001.json`: add reviewer name,
   score every dimension, and mark **each expected fact** met/partial/missed/
   contradicted. Include concrete examples of unsupported claims and citation
   errors. Use `notes` for any important tradeoffs.
3. Run `python -m scripts.evaluate_answers report`: it counts **only
   completely human-reviewed captured answers**; it never scores blank forms.

Do not publish selected evidence text or review data unless you intend to
publish the fictional corpus and have checked for secrets or sensitive content.

## Independent citation checks: what they can and cannot prove

Automated checks confirm whether citation IDs in `[Evidence N]` are valid,
whether the mapped chunk/document IDs match the supplied evidence, and whether
references are malformed. **They do not verify that the cited excerpt supports
the neighboring answer claim.** That is a human-reviewed rubric dimension.

If a response contains zero citations, the check reports
`has_citations=false`. A bare refusal of an unknowable exact fine may be
acceptable; an uncited affirmative policy claim is not.

## Quality improvement loop without cherry-picking

Categorize concrete failures before changing the pipeline:

| Failure | Diagnose first | Possible fix |
| --- | --- | --- |
| Missing document evidence | retrieval IDs and gold provenance | query wording, BM25/dense balance, chunking |
| Evidence was retrieved but not selected | context budget and diversity | context selection; avoid blindly increasing tokens |
| Evidence sufficient but answer fact missing | prompt and answer structure | instruct explicit coverage; make targeted test |
| Unsupported invented number | prompt + abstention | explicit uncertainty and unsupported-exact-value test |
| Bad citation | answer-to-evidence review | citation verification / stricter formatting |
| Poor relevance or organization | question vs generated answer | concise direct answer plus conditions/exceptions |

Keep the gold examples and failed answers. Re-evaluate each changed pipeline on
the **same** cases; avoid iteratively tuning to the final holdout. When
expanding toward 120 manually vetted cases, reserve a held-out question subset,
include paraphrases and novel combinations, and review table extraction
limitations. Do not call a metric "production quality" based on 30 questions.

## Release gate (provisional; not a percentage target)

Before a public "reliable" claim:
- end-to-end captures and independent human reviews cover the planned test set;
- critical fabricated policy/security claims and fabricated citations are
  investigated and resolved;
- unanswerable requests appropriately abstain;
- substantive claims point to supporting excerpts (not just valid IDs);
- failures under quota exhaustion and deployment instability are disclosed;
- real demo traffic is tested separately.

A single severe hallucination matters more than a polished average score.
