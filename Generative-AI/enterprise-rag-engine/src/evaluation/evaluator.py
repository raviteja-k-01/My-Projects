# src/evaluation/evaluator.py

import json
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run():
    path = PROJECT_ROOT / "data/eval/rag_outputs.json"

    with open(path) as f:
        data = json.load(f)

    dataset = Dataset.from_list(data)

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
    )

    print("\n===== RAGAS RESULTS =====\n")
    print(result)

    print(result.to_pandas())
if __name__ == "__main__":
    run()