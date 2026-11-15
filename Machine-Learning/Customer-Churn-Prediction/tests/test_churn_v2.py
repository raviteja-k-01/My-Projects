import joblib
import numpy as np
import pandas as pd
import pytest

from src.churn_v2 import build_pipeline, prepare_features, prepare_training_data, predict_probabilities


def sample_data(n=80):
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        churn = "Yes" if i % 3 == 0 else "No"
        rows.append({
            "customerID": f"TEST-{i}", "gender": "Male" if i % 2 else "Female",
            "SeniorCitizen": i % 2, "Partner": "Yes" if i % 4 else "No",
            "Dependents": "No", "tenure": i % 70, "PhoneService": "Yes",
            "MultipleLines": "No phone service" if i % 5 == 0 else "Yes",
            "InternetService": "DSL" if i % 2 else "Fiber optic",
            "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No",
            "TechSupport": "Yes", "StreamingTV": "No", "StreamingMovies": "Yes",
            "Contract": "Month-to-month" if i % 2 else "Two year",
            "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check" if i % 2 else "Mailed check",
            "MonthlyCharges": float(rng.uniform(25, 110)),
            "TotalCharges": " " if i == 4 else float(rng.uniform(40, 7000)),
            "Churn": churn,
        })
    return pd.DataFrame(rows)


def test_raw_features_correct_schema_and_no_target_leakage():
    df = sample_data()
    X, y = prepare_training_data(df)
    assert X.shape == (len(df), 19)
    assert "Churn" not in X.columns and "customerID" not in X.columns
    assert y.sum() == len(df[df.Churn == "Yes"])
    assert pd.isna(X.loc[4, "TotalCharges"])
    assert X.loc[0, "PaymentMethod"] == "Mailed check"


def test_missing_columns_rejected():
    with pytest.raises(ValueError, match="PaymentMethod"):
        prepare_features(sample_data().drop(columns=["PaymentMethod"]))


def test_pipeline_trains_and_reloaded_batch_matches_single(tmp_path):
    df = sample_data()
    X, y = prepare_training_data(df)
    pipe = build_pipeline().set_params(classifier__n_estimators=8)
    pipe.fit(X.iloc[:60], y.iloc[:60])
    single = predict_probabilities(pipe, X.iloc[[65]])
    batch = predict_probabilities(pipe, X.iloc[60:])
    assert 0 <= single[0] <= 1
    assert np.allclose(single[0], batch[5])
    path = tmp_path / "pipeline.joblib"
    joblib.dump(pipe, path)
    assert np.allclose(single, predict_probabilities(joblib.load(path), X.iloc[[65]]))


def test_preprocess_statistics_from_training_only():
    df = sample_data()
    X, y = prepare_training_data(df)
    X_train = X.iloc[:60].copy()
    X_test = X.iloc[60:].copy()
    X_test.loc[X_test.index[0], "MonthlyCharges"] = 999999.0
    pipe = build_pipeline().set_params(classifier__n_estimators=8)
    pipe.fit(X_train, y.iloc[:60])
    scaler = pipe.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scaler"]
    assert scaler.mean_[2] == pytest.approx(X_train["MonthlyCharges"].mean())
    assert np.isfinite(predict_probabilities(pipe, X_test)).all()
