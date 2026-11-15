"""Exercise Streamlit forms and rendered results against the prediction backend."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import src.preprocess as preprocessing
from src.preprocess import predict_batch, predict_probabilities

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.fixture
def app_model(tmp_path, monkeypatch, small_artifact):
    """Run the real app against an isolated, fitted model artifact."""
    path = tmp_path / "churn_pipeline.joblib"
    joblib.dump(small_artifact, path)
    monkeypatch.setattr(preprocessing, "MODEL_PATH", path)
    return small_artifact


def button(app, label):
    return next(item for item in app.button if item.label == label)


def assert_successful_run(app):
    assert not app.exception, [item.message for item in app.exception]
    assert not app.error, [item.value for item in app.error]


def test_single_prediction_matches_backend_and_shap_persists(app_model):
    app = AppTest.from_file(str(APP_PATH), default_timeout=40).run()
    assert_successful_run(app)
    app.selectbox(key="gender").select("Male")
    app.selectbox(key="Contract").select("Two year")
    app.number_input(key="tenure").set_value(24)
    app.number_input(key="MonthlyCharges").set_value(65.25)
    button(app, "Predict churn").click().run()
    assert_successful_run(app)

    saved = app.session_state["last_prediction"]
    probability = predict_probabilities(app_model, saved["raw"])[0]
    assert saved["result"]["Churn_Probability"].iloc[0] == probability
    assert app.metric[0].value == f"{probability:.2%}"
    expected_label = "Yes" if probability >= app_model["threshold"] else "No"
    assert app.success[0].value == f"Predicted churn: {expected_label}"
    assert saved["model_id"] == app_model["model_id"]

    button(app, "Explain prediction").click().run()
    assert_successful_run(app)
    assert len(app.get("imgs")) == 1
    assert app.get("imgs")[0].proto.imgs[0].url
    assert any("log-odds" in caption.value for caption in app.caption)

    app.run()
    assert_successful_run(app)
    assert app.metric[0].value == f"{probability:.2%}"
    assert len(app.get("imgs")) == 1

    app.sidebar.radio[0].set_value("Batch Prediction").run()
    assert_successful_run(app)
    assert len(app.get("file_uploader")) == 1
    assert any(
        item.proto.label == "Download sample CSV"
        for item in app.get("download_button")
    )
    app.sidebar.radio[0].set_value("Single Prediction").run()
    assert_successful_run(app)
    assert app.metric[0].value == f"{probability:.2%}"
    assert len(app.get("imgs")) == 1


def test_single_prediction_imputes_unknown_total_charges(app_model):
    app = AppTest.from_file(str(APP_PATH), default_timeout=40).run()
    app.checkbox(key="total_missing").check()
    button(app, "Predict churn").click().run()
    assert_successful_run(app)
    saved = app.session_state["last_prediction"]
    assert pd.isna(saved["raw"]["TotalCharges"].iloc[0])
    expected = predict_probabilities(app_model, saved["raw"])[0]
    assert np.isfinite(expected)
    assert saved["result"]["Churn_Probability"].iloc[0] == expected
    button(app, "Explain prediction").click().run()
    assert_successful_run(app)
    assert len(app.get("imgs")) == 1


def test_invalid_service_combination_shows_error_and_clears_old_result(app_model):
    app = AppTest.from_file(str(APP_PATH), default_timeout=40).run()
    button(app, "Predict churn").click().run()
    assert_successful_run(app)
    assert len(app.metric) == 1

    app.selectbox(key="PhoneService").select("No")
    app.selectbox(key="MultipleLines").select("Yes")
    button(app, "Predict churn").click().run()
    assert not app.exception
    assert len(app.error) == 1
    assert "MultipleLines" in app.error[0].value
    assert "PhoneService" in app.error[0].value
    assert len(app.metric) == 0
    assert len(app.success) == 0
    assert "last_prediction" not in app.session_state


def batch_harness(artifact, frame):
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from app.streamlit_app import render_batch\n"
        "render_batch(st.session_state['artifact'], st.session_state['customers'])\n",
        default_timeout=40,
    )
    app.session_state["artifact"] = artifact
    app.session_state["customers"] = frame
    return app.run()


@pytest.mark.parametrize("invalid_row", [False, True])
def test_batch_results_render_valid_records_and_errors(small_artifact, raw_customers, invalid_row):
    frame = raw_customers.head(3).copy()
    if invalid_row:
        frame.loc[1, "MonthlyCharges"] = -1
    from io import StringIO

    uploaded_frame = pd.read_csv(StringIO(frame.to_csv(index=False)))
    expected = predict_batch(small_artifact, uploaded_frame)
    app = batch_harness(small_artifact, frame)
    assert_successful_run(app)
    pd.testing.assert_frame_equal(app.dataframe[0].value, expected)
    valid_count = 2 if invalid_row else 3
    assert app.success[0].value == f"Predicted {valid_count} of 3 records."
    assert len(app.warning) == int(invalid_row)
    assert any(
        item.proto.label == "Download predictions"
        for item in app.get("download_button")
    )
    if invalid_row:
        assert pd.isna(app.dataframe[0].value.loc[1, "Churn_Probability"])
        assert "MonthlyCharges" in app.dataframe[0].value.loc[1, "Validation_Error"]


def test_batch_missing_column_shows_understandable_error(small_artifact, raw_customers):
    app = batch_harness(small_artifact, raw_customers.head(3).drop(columns=["Contract"]))
    assert not app.exception
    assert len(app.error) == 1
    assert "Missing required columns: Contract" in app.error[0].value
    assert len(app.dataframe) == 0
    assert len(app.get("download_button")) == 0
