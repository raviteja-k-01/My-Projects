"""Leakage-safe, reusable preprocessing and model for raw Telco Churn records."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLUMNS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert raw input to the stable schema; do not fit anything here."""
    missing = [name for name in FEATURE_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    cleaned = frame.loc[:, FEATURE_COLUMNS].copy()
    for name in NUMERIC_COLUMNS:
        cleaned[name] = pd.to_numeric(cleaned[name], errors="coerce")
    for name in CATEGORICAL_COLUMNS:
        cleaned[name] = cleaned[name].where(cleaned[name].notna(), np.nan).astype(object)
    return cleaned


def prepare_training_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if "Churn" not in frame.columns:
        raise ValueError("Missing target column: Churn")
    label = frame["Churn"].astype("string").str.strip().map({"No": 0, "Yes": 1})
    if label.isna().any():
        raise ValueError("Churn must contain only 'Yes' or 'No'")
    return prepare_features(frame), label.astype(int)


def build_pipeline() -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocess = ColumnTransformer(
        [("numeric", numeric, NUMERIC_COLUMNS),
         ("categorical", categorical, CATEGORICAL_COLUMNS)],
        remainder="drop", verbose_feature_names_out=False,
    )
    model = XGBClassifier(
        n_estimators=250, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="logloss", n_jobs=2,
    )
    return Pipeline([("preprocess", preprocess), ("classifier", model)])


def predict_probabilities(pipeline: Pipeline, raw_frame: pd.DataFrame) -> np.ndarray:
    return pipeline.predict_proba(prepare_features(raw_frame))[:, 1]
