# Archived experiments — not supported v1 commands

These exact historical sources used the removed `pipeline.reranker` and
`DEFAULT_RERANK_LIMIT` interfaces. The current v1 serving path uses hybrid
retrieval and context selection without a cross-encoder. The A/B experiment
also invokes an LLM judge; it is not the human-review evaluation protocol.

Sources are retained as `.py.txt` files for reference. The former command paths
now exit with an explanatory message before initializing any model or service.
Restoring an experiment requires a separately scoped design, optional model
dependencies, explicit API budgets and independent validation.

Supported workflow: `python -m scripts.evaluate_answers status`, `capture`,
`inspect`, and `report`. See `docs/ENGINEERING_STATUS.md` for corpus arguments.
