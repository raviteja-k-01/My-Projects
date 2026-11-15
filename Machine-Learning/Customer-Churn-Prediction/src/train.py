"""Train once on distinct customers and evaluate an untouched stratified holdout."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import tempfile
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .preprocess import (
    PROJECT_DIR, RANDOM_STATE, RAW_DATA, THRESHOLD, build_pipeline,
    load_artifact, load_dataset, predict_probabilities, prepare_training_data,
)


def split_dataset(frame: pd.DataFrame):
    if frame["customerID"].duplicated().any():
        raise ValueError("Split requires one record per customerID; call load_dataset first")
    X, y = prepare_training_data(frame)
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


def calculate_metrics(y_true, probabilities, threshold: float = THRESHOLD) -> dict:
    predicted = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "precision_churn": float(precision_score(y_true, predicted, pos_label=1, zero_division=0)),
        "recall_churn": float(recall_score(y_true, predicted, pos_label=1, zero_division=0)),
        "f1_churn": float(f1_score(y_true, predicted, pos_label=1, zero_division=0)),
        "confusion_matrix_no_yes": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(),
    }


def train(raw_data: Path = RAW_DATA, output_dir: Path = PROJECT_DIR) -> dict:
    frame, profile = load_dataset(raw_data)
    X_train, X_test, y_train, y_test = split_dataset(frame)
    train_ids = set(frame.loc[X_train.index, "customerID"])
    test_ids = set(frame.loc[X_test.index, "customerID"])
    if train_ids & test_ids:
        raise ValueError("A customer appears in both training and test sets")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    probability = pipeline.predict_proba(X_test)[:, 1]
    metrics = calculate_metrics(y_test, probability)
    versions = {name: importlib.metadata.version(name) for name in (
        "numpy", "pandas", "scikit-learn", "xgboost", "joblib", "shap", "streamlit", "matplotlib",
    )}
    source_hashes = {name: hashlib.sha256((PROJECT_DIR / "src" / name).read_bytes()).hexdigest()
                     for name in ("preprocess.py", "train.py")}
    # Omit unset defaults and the native NaN missing-value marker from JSON metadata.
    params = {key: value for key, value in pipeline.named_steps["classifier"].get_params().items()
              if value is not None and not (isinstance(value, float) and not np.isfinite(value))}
    identity = {"data_sha256": profile["sha256"], "sources": source_hashes,
                "parameters": params, "versions": versions, "threshold": THRESHOLD,
                "split": "deduplicate-exact-then-stratified-80-20-seed-42"}
    model_id = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]
    report = {
        "model_id": model_id, "positive_class": "Yes (churn)",
        "target_encoding": {"No": 0, "Yes": 1}, "decision_threshold": THRESHOLD,
        "random_state": RANDOM_STATE, "train_rows": len(X_train), "test_rows": len(X_test),
        "train_target_counts": {"No": int((y_train == 0).sum()), "Yes": int((y_train == 1).sum())},
        "test_target_counts": {"No": int((y_test == 0).sum()), "Yes": int((y_test == 1).sum())},
        "overlapping_customers": len(train_ids & test_ids),
        "evaluation_method": "Exact duplicate removal; stratified 80/20 customer holdout; no tuning",
        "threshold_selection": "Fixed 0.50 before evaluation; no threshold search",
        "dataset": profile, "python_version": platform.python_version(),
        "package_versions": versions, "source_sha256": source_hashes,
        "model_parameters": params, **metrics,
    }
    artifact = {"pipeline": pipeline, "threshold": THRESHOLD, "model_id": model_id,
                "metadata": report.copy()}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Stage and reload before replacing the canonical model or reports.
    with tempfile.TemporaryDirectory(prefix=".training-", dir=output_dir) as temporary:
        stage = Path(temporary)
        model_path = stage / "churn_pipeline.joblib"
        joblib.dump(artifact, model_path, compress=3)
        reloaded = load_artifact(model_path)
        after = predict_probabilities(reloaded, X_test)
        if not np.array_equal(probability, after):
            raise RuntimeError("Serialization changed held-out predictions")
        if calculate_metrics(y_test, after) != metrics:
            raise RuntimeError("Reloaded model metrics differ")
        report["serialization_predictions_identical"] = True
        report["artifact_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()

        split = frame[["customerID", "Churn"]].copy()
        split.insert(0, "source_row", frame.index)
        split["partition"] = np.where(split.index.isin(X_train.index), "train", "test")
        split.to_csv(stage / "split.csv", index=False)
        report["split_sha256"] = hashlib.sha256((stage / "split.csv").read_bytes()).hexdigest()
        test_results = frame.loc[X_test.index, ["customerID"]].copy()
        test_results.insert(0, "source_row", X_test.index)
        test_results["actual_churn"] = y_test
        test_results["Churn_Probability"] = probability.astype(float)
        test_results["Predicted_Churn"] = (probability >= THRESHOLD).astype(int)
        test_results.to_csv(stage / "test_predictions.csv", index=False, float_format="%.17g")
        (stage / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (stage / "dataset_profile.json").write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")

        ConfusionMatrixDisplay(np.array(metrics["confusion_matrix_no_yes"]),
                               display_labels=["No", "Yes"]).plot(cmap="Blues", colorbar=False)
        plt.title("Distinct-customer holdout; threshold 0.50")
        plt.tight_layout()
        plt.savefig(stage / "confusion_matrix.png", dpi=140)
        plt.close()
        names = pipeline.named_steps["preprocess"].get_feature_names_out()
        importance = pipeline.named_steps["classifier"].feature_importances_
        order = np.argsort(importance)[-15:]
        plt.figure(figsize=(9, 6))
        plt.barh(names[order], importance[order])
        plt.xlabel("XGBoost feature importance (normalized gain)")
        plt.tight_layout()
        plt.savefig(stage / "feature_importance.png", dpi=140)
        plt.close()

        (output_dir / "model").mkdir(exist_ok=True)
        (output_dir / "reports").mkdir(exist_ok=True)
        model_path.replace(output_dir / "model" / model_path.name)
        for path in stage.iterdir():
            path.replace(output_dir / "reports" / path.name)
    print(json.dumps({key: report[key] for key in (
        "model_id", "train_rows", "test_rows", "accuracy", "roc_auc", "precision_churn",
        "recall_churn", "f1_churn", "confusion_matrix_no_yes", "serialization_predictions_identical",
    )}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=RAW_DATA)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR)
    args = parser.parse_args()
    train(args.data, args.output_dir)
