"""Small raw customer fixtures for fast pipeline and application checks."""

from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import build_pipeline, prepare_training_data


@pytest.fixture
def raw_customers():
    rows = []
    for i in range(80):
        rows.append({
            "customerID": f"TEST-{i:04d}",
            "gender": "Female" if i % 2 else "Male",
            "SeniorCitizen": i % 2,
            "Partner": "Yes" if i % 4 else "No",
            "Dependents": "No",
            "tenure": i % 70,
            "PhoneService": "Yes",
            "MultipleLines": "Yes" if i % 2 else "No",
            "InternetService": "DSL" if i % 2 else "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month" if i % 2 else "Two year",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check" if i % 2 else "Mailed check",
            "MonthlyCharges": 25.0 + i * 0.75,
            "TotalCharges": " " if i == 4 else float(40 + i * 71),
            "Churn": "Yes" if i % 3 == 0 else "No",
        })
    return pd.DataFrame(rows)


@pytest.fixture
def small_artifact(raw_customers):
    _, target = prepare_training_data(raw_customers)
    pipeline = build_pipeline().set_params(classifier__n_estimators=8)
    pipeline.fit(raw_customers.iloc[:60], target.iloc[:60])
    return {
        "pipeline": pipeline,
        "threshold": 0.5,
        "model_id": "test-model",
        "metadata": {},
    }
