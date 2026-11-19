# Development label observations (assistant review, not human accuracy)

Original dataset SHA-256:
`040520cd1c25a006216e345cff6001b696838c68522421a40205a205331b9086`.
No dataset facts, reference answers, or historical responses were changed.
These are proposed review distinctions, not adopted scoring labels.

## E001: public AI data

The three expected facts (PUBLIC information, low-risk synthetic/test content,
explicit approved exception) directly answer the question. Treat each as a
required fact, subject to independent source review. The saved answer states
all three and cites `AI-GOV-05-B0035`, page 24, which contains that wording.

Its additional prohibition list cites `IAM-STD-03-B0130`, page 94. That excerpt
prohibits credentials, client data, CONFIDENTIAL and RESTRICTED information.
The AI governance excerpt also states that RESTRICTED information is never
permitted in public AI. The response should not be interpreted as establishing
that the exception clause overrides the IAM prohibitions: the excerpts do not
resolve that authority question. Review this ambiguity if evaluating exceptions.
This observation does not certify every claim or assign a score.

## E002: enterprise AI with CONFIDENTIAL documents

The question asks whether CONFIDENTIAL document use is allowed and what must
be in place first. `AI-GOV-05-B0036`, page 24, states the approved use case and
service configuration requirements. Those two expected facts are required.

The other expected fact, “Enterprise AI Services may handle INTERNAL information
by default,” is useful background about a different data classification. It
should be considered optional for this question, with a reviewer rationale,
rather than automatically marking an otherwise sufficient confidential-data
answer incomplete. The saved answer omits that background but states both
confidential-data approvals. No original label was edited to improve a score.

The longer answer also cites governance and privacy obligations in
`AI-GOV-05-B0029` (page 20), `AI-GOV-05-B0038` (page 26), and
`PRIV-DG-02-B0093` (page 77). These must be checked claim by claim in a real
review; a valid citation ID alone does not prove support.

## Remaining review work

The 80-question development set contains 101 supplied evidence snippets.
The offline before/after diagnostic matches 73 snippets in selected evidence
in both configurations. These snippets are search anchors, not independently
verified exhaustive relevance labels. Missing anchors can reflect retrieval,
selection, paraphrasing, or label quality; do not automatically assign blame.
Keep required/optional fact decisions, relevance labels, and semantic reviews
in separate source-locked annotation files. Do not modify the held-out set.
