"""Shared raw-input validation and train-only preprocessing for Telco customers."""

from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from xgboost import XGBClassifier

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_DIR / "data" / "telecom.csv"
MODEL_PATH = PROJECT_DIR / "model" / "churn_pipeline.joblib"
RANDOM_STATE = 42
THRESHOLD = 0.5
NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_VALUES = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    **{name: ["No", "Yes", "No internet service"] for name in [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies",
    ]},
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": ["Electronic check", "Mailed check",
                      "Bank transfer (automatic)", "Credit card (automatic)"],
}
CATEGORICAL_COLUMNS = list(CATEGORICAL_VALUES)
FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]


def read_customer_csv(source) -> pd.DataFrame:
    """Read a UTF-8 CSV without hiding duplicate headers or malformed row lengths."""
    if hasattr(source, "read"):
        content = source.read()
    else:
        content = Path(source).read_bytes()
    if isinstance(content, bytes):
        content = content.decode("utf-8-sig")
    try:
        rows = csv.reader(io.StringIO(content), strict=True)
        header = next(rows)
        if not header or any(not name.strip() for name in header):
            raise ValueError("CSV must have named columns.")
        if len(set(header)) != len(header):
            raise ValueError("Duplicate CSV column names are not allowed.")
        for row_number, row in enumerate(rows, start=2):
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(f"CSV row {row_number} has {len(row)} fields; expected {len(header)}.")
    except StopIteration as exc:
        raise ValueError("CSV is empty.") from exc
    except csv.Error as exc:
        raise ValueError(f"Malformed CSV: {exc}") from exc
    # Only blank cells are missing. Text such as 'NA' in a numeric field is invalid.
    return pd.read_csv(io.StringIO(content), keep_default_na=False)


def validate_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return normalized features and per-row errors, without learning statistics."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Customer input must be a table with named columns.")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate input column names are not allowed.")
    missing = [name for name in FEATURE_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("Customer input contains no records.")
    cleaned = frame.loc[:, FEATURE_COLUMNS].copy()
    errors = np.full(len(frame), "", dtype=object)

    def flag(mask, message):
        for position in np.flatnonzero(np.asarray(mask)):
            errors[position] += ("; " if errors[position] else "") + message

    for name in FEATURE_COLUMNS:
        values = cleaned[name].astype(object).map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
        missing_values = values.isna() | values.eq("")
        if name in NUMERIC_COLUMNS:
            numeric = pd.to_numeric(values.mask(missing_values, np.nan), errors="coerce")
            flag(~missing_values & (numeric.isna() | ~np.isfinite(numeric)),
                 f"{name} must be a finite number or blank")
            flag(numeric < 0, f"{name} must be non-negative")
            if name == "SeniorCitizen":
                flag(numeric.notna() & ~numeric.isin([0, 1]), "SeniorCitizen must be 0 or 1")
            if name == "tenure":
                flag(numeric.notna() & np.isfinite(numeric) & (numeric % 1 != 0),
                     "tenure must be a whole number of months")
            cleaned[name] = numeric.astype(float)
        else:
            flag(~missing_values & ~values.isin(CATEGORICAL_VALUES[name]),
                 f"{name} must be one of: {', '.join(CATEGORICAL_VALUES[name])}")
            cleaned[name] = values.mask(missing_values, np.nan).astype(object)
    phone, lines = cleaned["PhoneService"], cleaned["MultipleLines"]
    both_phone = phone.notna() & lines.notna()
    flag(both_phone & (phone.eq("No") != lines.eq("No phone service")),
         "MultipleLines must be 'No phone service' exactly when PhoneService is 'No'")
    internet = cleaned["InternetService"]
    for name in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                 "StreamingTV", "StreamingMovies"]:
        both_internet = internet.notna() & cleaned[name].notna()
        flag(both_internet & (internet.eq("No") != cleaned[name].eq("No internet service")),
             f"{name} must be 'No internet service' exactly when InternetService is 'No'")
    return cleaned, pd.Series(errors, index=frame.index, name="Validation_Error")


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned, errors = validate_features(frame)
    if errors.ne("").any():
        messages = [f"Record {position + 1}: {message}" for position, message in
                    enumerate(errors) if message]
        raise ValueError("Invalid customer input. " + " | ".join(messages[:5]))
    return cleaned


def prepare_training_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if "Churn" not in frame.columns:
        raise ValueError("Missing target column: Churn")
    label = frame["Churn"].astype("string").str.strip().map({"No": 0, "Yes": 1})
    if label.isna().any():
        raise ValueError("Churn must contain only 'Yes' or 'No', without missing targets")
    return prepare_features(frame), label.astype(int)


def load_dataset(path: Path = RAW_DATA) -> tuple[pd.DataFrame, dict]:
    """Audit raw data and remove exact repeats before any customer-level split."""
    path = Path(path)
    frame = read_customer_csv(path)
    features, target = prepare_training_data(frame)
    if "customerID" not in frame:
        raise ValueError("Missing customerID column; cannot verify customer separation")
    ids = frame["customerID"].astype("string").str.strip()
    if ids.isna().any() or ids.eq("").any():
        raise ValueError("customerID must not be missing")
    frame = frame.copy()
    frame["customerID"] = ids.astype(object)
    unique = frame.drop_duplicates().copy()
    if unique["customerID"].duplicated().any():
        raise ValueError("Conflicting records for the same customerID; resolve before training")
    if target.nunique() != 2:
        raise ValueError("Training data must contain both Churn classes: No and Yes")
    profile = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "raw_rows": len(frame), "raw_columns": len(frame.columns),
        "feature_columns": FEATURE_COLUMNS,
        "missing_features": {name: int(features[name].isna().sum()) for name in FEATURE_COLUMNS},
        "raw_target_counts": {str(k): int(v) for k, v in frame["Churn"].value_counts().items()},
        "exact_duplicates_removed": len(frame) - len(unique),
        "unique_customers": len(unique),
        "unique_target_counts": {str(k): int(v) for k, v in unique["Churn"].value_counts().items()},
    }
    return unique, profile


def build_pipeline() -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True))])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC_COLUMNS),
        ("categorical", categorical, CATEGORICAL_COLUMNS),
    ], remainder="drop", verbose_feature_names_out=False)
    model = XGBClassifier(
        n_estimators=250, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, random_state=RANDOM_STATE, eval_metric="logloss",
        n_jobs=2, tree_method="hist",
    )
    # Cleaning is stateless; fit learns imputation and encoding on training rows only.
    return Pipeline([
        ("schema", FunctionTransformer(prepare_features, validate=False)),
        ("preprocess", preprocess), ("classifier", model),
    ])


def load_artifact(path: Path = MODEL_PATH) -> dict:
    """Load a trusted, locally trained artifact; never load joblib from strangers."""
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or not {"pipeline", "threshold", "model_id", "metadata"} <= artifact.keys():
        raise ValueError("Invalid model artifact. Run python -m src.train.")
    if not 0 < artifact["threshold"] < 1:
        raise ValueError("Model threshold must be between 0 and 1")
    if not np.array_equal(artifact["pipeline"].classes_, [0, 1]):
        raise ValueError("Model classes must be No=0 and Yes=1")
    return artifact


def predict_probabilities(artifact: dict, raw_frame: pd.DataFrame) -> np.ndarray:
    return artifact["pipeline"].predict_proba(raw_frame)[:, 1]


def predict_batch(artifact: dict, raw_frame: pd.DataFrame) -> pd.DataFrame:
    """Keep invalid records with errors while predicting every valid record."""
    cleaned, errors = validate_features(raw_frame)
    valid = errors.eq("").to_numpy()
    probabilities = np.full(len(raw_frame), np.nan)
    labels = np.full(len(raw_frame), "", dtype=object)
    if valid.any():
        probabilities[valid] = predict_probabilities(artifact, cleaned.iloc[np.flatnonzero(valid)])
        labels[valid] = np.where(probabilities[valid] >= artifact["threshold"], "Yes", "No")
    result = raw_frame.copy()
    result["Predicted_Churn"] = labels
    result["Churn_Probability"] = probabilities
    result["Validation_Error"] = errors.to_numpy()
    return result
