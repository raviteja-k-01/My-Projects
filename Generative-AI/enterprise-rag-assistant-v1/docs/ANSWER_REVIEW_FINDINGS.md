# Preliminary source comparison — 2026-11-19

This is an assistant's inspection of the five existing saved answers against
their **actual selected evidence**, not an independent human review. No review
forms, expected facts, answers, ratings or fact-check statuses were changed.
No LLM-judge API was used. All five answers pass the existing mechanical
citation checks and their selected evidence matches the verified corpus.
Those properties alone do not certify factual correctness.

| Question | Source comparison | Finding requiring review |
| --- | --- | --- |
| Q001 | The answer includes the timestamp, reason and approving role. All appear in Evidence 1, `SEC-IR-004-B0026`, page 16. | No specific omission found in this inspection; independent review remains required. |
| Q002 | The named commander and executive-group activation conditions appear in Evidence 1 (`SEC-IR-004-B0044`, page 26) and Evidence 2 (`SEC-IR-004-B0098`, page 59). | **The answer omits the expected fact that functional specialists retain domain authority within the common command structure.** Both supplied excerpts contain this fact. This is an answer-completeness finding, not a retrieval failure. |
| Q003 | Both protecting credentials/devices/information and prompt reporting appear in Evidence 1, `SEC-POL-01-B0116`, page 93. | The exact gold chunk is a different, overlapping chunk. Missing the gold ID from selected context is not evidence that the facts were absent. |
| Q004 | The six expected MFA/password rules and the extra break-glass/MFA-prompt instructions appear in Evidence 1, `IAM-STD-03-B0037`, page 30. | The selected source differs from the gold ID but contains the relevant policy text. Independently check conditional wording and claim-to-source support. |
| Q005 | The four expected corporate-card facts appear in Evidence 2 (`FIN-EXP-12-B0117`, page 85) and Evidence 4 (`FIN-EXP-12-B0077`, page 59). The added procurement, conflict-of-interest, missing-evidence and exception statements are present in the cited excerpts. | Answer is longer than the core question requires. A human should assess relevance and policy scope, especially the handbook exception wording from Evidence 1, `HR-HBK-01-B0049`, page 48. |

## Why a source-ID metric cannot replace review

The expected chunks for Q001–Q005 are present at saved retrieval ranks
1, 2, 2, 3 and 6 respectively. Context deduplication selects other overlapping
chunks for Q002–Q004. Q002's omitted fact is still present in the selected
context; Q003 and Q004 convey the expected facts from other selected chunks.
Changing the gold references or ranking algorithm to improve this diagnostic
would not be justified by these observations.

## Remaining cases

Q006's first new embedding attempt returned HTTP 429; it produced no saved
retrieval or answer. Q007–Q030 were not attempted after that failure. None of
those 25 questions can be assessed for generated-answer quality yet.
Q026–Q028 are the unanswerable cases and must eventually be checked for honest
abstention against retrieved evidence. Q027 alone has no expected retrieval
chunk; this does not exempt it from answer evaluation.

The generated local `data/processed/answer_review_packet.md` includes all 30
questions, 121 expected facts, saved answers, the actual supplied evidence,
and gold-referenced source excerpts. Review each fact and claim there, then
fill the existing `data/processed/answer_review_forms/Qxxx.json` files.
The current human-reviewed count remains **0/30**.
