"""Explain the fitted classifier using its actual transformed inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .preprocess import MODEL_PATH, load_artifact, prepare_features

PROJECT_DIR = Path(__file__).resolve().parents[1]


def explain_predictions(artifact: dict, raw: pd.DataFrame) -> shap.Explanation:
    """Return feature contributions to the Churn=Yes log-odds.

    Tree-path-dependent SHAP uses the fitted trees' training path counts.
    The input to the explainer is exactly the classifier's encoded input.
    """
    pipeline = artifact["pipeline"]
    preprocessor = pipeline.named_steps["preprocess"]
    transformed = preprocessor.transform(prepare_features(raw))
    classifier = pipeline.named_steps["classifier"]
    names = preprocessor.get_feature_names_out().tolist()
    explainer = shap.TreeExplainer(
        classifier, model_output="raw", feature_perturbation="tree_path_dependent",
        feature_names=names,
    )
    explanation = explainer(transformed, check_additivity=True)
    margin = classifier.predict(transformed, output_margin=True)
    reconstructed = explanation.base_values + explanation.values.sum(axis=1)
    if not np.allclose(reconstructed, margin, atol=1e-5, rtol=1e-5):
        raise ValueError("SHAP contributions do not reconstruct the model output.")
    explanation.feature_names = names
    return explanation


def waterfall_figure(explanation: shap.Explanation, index: int = 0):
    """Render one explanation; the caller owns and closes the figure."""
    shap.plots.waterfall(explanation[index], max_display=12, show=False)
    figure = plt.gcf()
    figure.suptitle("Feature contributions to churn log-odds", y=1.02)
    return figure


def generate_reports(
    model_path: Path = MODEL_PATH,
    data_path: Path = PROJECT_DIR / "data" / "telecom.csv",
    output_dir: Path = PROJECT_DIR / "reports",
    max_rows: int = 200,
) -> dict:
    """Explain a fixed, disclosed sample without fitting or tuning the model."""
    if max_rows < 1:
        raise ValueError("max_rows must be positive.")
    artifact = load_artifact(model_path)
    raw = pd.read_csv(data_path).drop_duplicates().head(max_rows)
    values = explain_predictions(artifact, raw)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure = waterfall_figure(values)
    figure.savefig(output_dir / "shap_local_explain.png", bbox_inches="tight", dpi=140)
    plt.close(figure)

    shap.plots.bar(values, max_display=15, show=False)
    figure = plt.gcf()
    figure.suptitle(f"Mean absolute SHAP contribution ({len(raw)} customers)", y=1.02)
    figure.savefig(output_dir / "shap_global_importance.png", bbox_inches="tight", dpi=140)
    plt.close(figure)

    summary = {
        "model_id": artifact["model_id"],
        "rows_explained": len(raw),
        "sample": f"First {max_rows} exact-deduplicated rows in the input CSV",
        "output_units": "Churn=Yes log-odds",
        "method": "TreeExplainer, tree_path_dependent",
        "local_plot_row": 0,
        "additivity_verified": True,
        "note": "Descriptive explanations of this model, not causal effects or evaluation metrics.",
    }
    (output_dir / "shap_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--data", type=Path, default=PROJECT_DIR / "data" / "telecom.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "reports")
    parser.add_argument("--max-rows", type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(generate_reports(args.model, args.data, args.output_dir, args.max_rows), indent=2))


if __name__ == "__main__":
    main()
