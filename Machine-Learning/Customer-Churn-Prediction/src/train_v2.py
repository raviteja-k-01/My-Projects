"""Train/evaluate on the RAW dataset, with preprocessing fitted on train only.

Usage (from Customer-Churn-Prediction project directory):
    python -m src.train_v2
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .churn_v2 import build_pipeline, prepare_training_data

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_DIR / "data" / "telecom.csv"
MODEL_OUT = PROJECT_DIR / "model" / "churn_pipeline_v2.joblib"
METRICS_OUT = PROJECT_DIR / "reports" / "churn_metrics_v2.json"
THRESHOLD = 0.50


def train(raw_data: Path = RAW_DATA, model_out: Path = MODEL_OUT,
          metrics_out: Path = METRICS_OUT) -> dict:
    frame = pd.read_csv(raw_data)
    X, y = prepare_training_data(frame)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )

    # The full preprocessor is inside this pipeline. Its fit() sees X_train only.
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    probability = pipeline.predict_proba(X_test)[:, 1]
    predicted = (probability >= THRESHOLD).astype(int)
    metrics = {
        "positive_class": "Yes (churn)",
        "decision_threshold": THRESHOLD,
        "random_state": 42,
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, predicted)),
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "precision_churn": float(precision_score(y_test, predicted, zero_division=0)),
        "recall_churn": float(recall_score(y_test, predicted, zero_division=0)),
        "f1_churn": float(f1_score(y_test, predicted, zero_division=0)),
        "confusion_matrix_no_yes": confusion_matrix(y_test, predicted, labels=[0, 1]).tolist(),
    }
    model_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_out)
    metrics_out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"New model saved: {model_out}")
    print(f"New metrics saved: {metrics_out}")
    return metrics


if __name__ == "__main__":
    train()
