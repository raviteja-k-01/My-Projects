"""Customer Churn v2: RAW inputs processed by the fitted training pipeline.

Run `python -m src.train_v2` before starting this app.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.churn_v2 import prepare_features, predict_probabilities  # noqa: E402

MODEL_PATH = ROOT / "model" / "churn_pipeline_v2.joblib"
THRESHOLD = 0.50

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("Customer Churn Prediction — corrected pipeline")

if not MODEL_PATH.exists():
    st.error("Corrected model missing. Run `python -m src.train_v2` first.")
    st.stop()


@st.cache_resource
def load_pipeline(path: str):
    # Only load your own locally generated joblib file, never an untrusted download.
    return joblib.load(path)


pipeline = load_pipeline(str(MODEL_PATH))
mode = st.sidebar.radio("Mode", ["Single Prediction", "Batch Prediction"])


def show_predictions(frame: pd.DataFrame):
    probabilities = predict_probabilities(pipeline, frame)
    predictions = probabilities >= THRESHOLD
    result = frame.copy()
    result["Predicted_Churn"] = np.where(predictions, "Yes", "No")
    result["Churn_Probability"] = np.round(probabilities, 4)
    return result


if mode == "Single Prediction":
    with st.form("customer"):
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", 0, 100, 12)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        options = ["No", "Yes", "No internet service"]
        security = st.selectbox("Online Security", options)
        backup = st.selectbox("Online Backup", options)
        protection = st.selectbox("Device Protection", options)
        support = st.selectbox("Tech Support", options)
        tv = st.selectbox("Streaming TV", options)
        movies = st.selectbox("Streaming Movies", options)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly = st.number_input("Monthly Charges", 0.0, 1000.0, 70.0)
        total = st.number_input("Total Charges", 0.0, 100000.0, 2000.0)
        submitted = st.form_submit_button("Predict")
    if submitted:
        raw = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
            "MultipleLines": multiple, "InternetService": internet,
            "OnlineSecurity": security, "OnlineBackup": backup,
            "DeviceProtection": protection, "TechSupport": support,
            "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": total,
        }])
        st.session_state["churn_last_raw"] = raw
    if "churn_last_raw" in st.session_state:
        raw = st.session_state["churn_last_raw"]
        result = show_predictions(raw)
        st.dataframe(result[["Predicted_Churn", "Churn_Probability"]], hide_index=True)
        if st.button("Explain prediction with SHAP"):
            try:
                import shap
                import matplotlib.pyplot as plt
                X = pipeline.named_steps["preprocess"].transform(prepare_features(raw))
                explainer = shap.TreeExplainer(pipeline.named_steps["classifier"])
                values = explainer(X)
                feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
                exp = shap.Explanation(values=values.values[0], base_values=values.base_values[0],
                                       data=X[0], feature_names=feature_names)
                shap.plots.waterfall(exp, max_display=12, show=False)
                st.pyplot(plt.gcf(), clear_figure=True)
            except Exception as exc:
                st.warning(f"SHAP explanation unavailable: {exc}")

else:
    file = st.file_uploader("Upload a RAW customer CSV (not the old scaled CSV)", type=["csv"])
    if file is not None:
        raw = pd.read_csv(file)
        try:
            result = show_predictions(raw)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.dataframe(result.head(50))
            st.download_button("Download predictions", result.to_csv(index=False).encode("utf-8"),
                               "churn_predictions_v2.csv", "text/csv")
