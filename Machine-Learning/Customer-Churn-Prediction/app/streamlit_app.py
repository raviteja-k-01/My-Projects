"""Single and batch churn predictions from raw customer information."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import (  # noqa: E402
    CATEGORICAL_VALUES,
    MODEL_PATH,
    load_artifact,
    predict_batch,
    read_customer_csv,
)

LABELS = {
    "gender": "Gender", "Partner": "Partner", "Dependents": "Dependents",
    "PhoneService": "Phone service", "MultipleLines": "Multiple lines",
    "InternetService": "Internet service", "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup", "DeviceProtection": "Device protection",
    "TechSupport": "Tech support", "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies", "Contract": "Contract",
    "PaperlessBilling": "Paperless billing", "PaymentMethod": "Payment method",
}


@st.cache_resource
def cached_artifact(path: str, modified_ns: int, size: int) -> dict:
    """Refresh the cached artifact when a training run replaces the file."""
    return load_artifact(Path(path))


def render_batch(artifact: dict, frame: pd.DataFrame) -> pd.DataFrame | None:
    """Render validated row results and the exact downloadable CSV."""
    try:
        result = predict_batch(artifact, frame)
    except ValueError as exc:
        st.error(str(exc))
        return None
    valid = result["Validation_Error"].fillna("").eq("")
    st.success(f"Predicted {int(valid.sum())} of {len(result)} records.")
    if not valid.all():
        st.warning(
            f"{int((~valid).sum())} invalid records were left without predictions. "
            "Check Validation_Error for the fields to correct."
        )
    st.dataframe(result, hide_index=True, use_container_width=True)
    st.download_button(
        "Download predictions",
        result.to_csv(index=False).encode("utf-8"),
        file_name="churn_predictions.csv", mime="text/csv",
    )
    return result


def single_prediction(artifact: dict) -> None:
    st.subheader("Enter customer details")
    with st.form("customer"):
        raw = {}
        for column, choices in CATEGORICAL_VALUES.items():
            default_index = choices.index("Yes") if column == "PhoneService" else 0
            raw[column] = st.selectbox(LABELS[column], choices, index=default_index, key=column)
        raw["SeniorCitizen"] = st.selectbox("Senior citizen", [0, 1], key="SeniorCitizen")
        raw["tenure"] = st.number_input("Tenure (months)", min_value=0, value=12, step=1, key="tenure")
        raw["MonthlyCharges"] = st.number_input(
            "Monthly charges", min_value=0.0, value=70.0, step=0.05, key="MonthlyCharges"
        )
        raw["TotalCharges"] = st.number_input(
            "Total charges", min_value=0.0, value=2000.0, step=0.05, key="TotalCharges"
        )
        total_missing = st.checkbox("Total charges are unknown", key="total_missing")
        st.caption("If there is no phone or internet service, select the matching 'No service' options for its add-ons.")
        submitted = st.form_submit_button("Predict churn")

    if submitted:
        if total_missing:
            raw["TotalCharges"] = np.nan
        frame = pd.DataFrame([raw])
        try:
            result = predict_batch(artifact, frame)
            error = result["Validation_Error"].fillna("").iloc[0]
            if error:
                raise ValueError(error)
        except ValueError as exc:
            st.session_state.pop("last_prediction", None)
            st.error(str(exc))
        else:
            st.session_state["last_prediction"] = {
                "raw": frame, "result": result, "model_id": artifact["model_id"],
                "show_explanation": False,
            }

    previous = st.session_state.get("last_prediction")
    if previous is None or previous["model_id"] != artifact["model_id"]:
        return
    result = previous["result"].iloc[0]
    st.success(f"Predicted churn: {result['Predicted_Churn']}")
    st.metric("Churn probability", f"{result['Churn_Probability']:.2%}")
    st.caption("Results below use the most recently submitted customer details.")
    with st.expander("Customer details used for this prediction"):
        st.dataframe(previous["raw"], hide_index=True)
    if st.button("Explain prediction"):
        previous["show_explanation"] = True
    if previous["show_explanation"]:
        from src.explain import explain_predictions, waterfall_figure
        import matplotlib.pyplot as plt

        try:
            explanation = explain_predictions(artifact, previous["raw"])
            figure = waterfall_figure(explanation)
            try:
                st.pyplot(figure, clear_figure=True)
            finally:
                plt.close(figure)
            st.caption(
                "SHAP contributions are in log-odds: positive values increase the model's "
                "churn score; negative values decrease it. These are model explanations, "
                "not causal effects."
            )
        except (ValueError, RuntimeError) as exc:
            st.error(f"Could not explain this prediction: {exc}")


def batch_prediction(artifact: dict) -> None:
    st.subheader("Upload customer information")
    st.write(
        "Use the 19 raw customer fields shown in the sample CSV. "
        "Blank values are filled using the training data; invalid values are flagged. "
        "customerID and Churn are optional and are not used to make predictions."
    )
    sample_path = ROOT / "data" / "telecom.csv"
    if sample_path.exists():
        sample = pd.read_csv(sample_path, nrows=5).drop(columns=["Churn"], errors="ignore")
        st.download_button(
            "Download sample CSV", sample.to_csv(index=False).encode("utf-8"),
            file_name="sample_customers.csv", mime="text/csv",
        )
    uploaded = st.file_uploader("Upload a customer CSV", type=["csv"])
    if uploaded is not None:
        try:
            frame = read_customer_csv(uploaded)
        except (ValueError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            st.error(f"Could not read customer CSV: {exc}")
        else:
            render_batch(artifact, frame)


def main() -> None:
    st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
    st.title("Customer Churn Prediction")
    if not MODEL_PATH.exists():
        st.error("Model missing. Run `python -m src.train` from the project folder first.")
        st.stop()
    try:
        stat = MODEL_PATH.stat()
        artifact = cached_artifact(str(MODEL_PATH), stat.st_mtime_ns, stat.st_size)
    except (ValueError, OSError, KeyError) as exc:
        st.error(f"Could not load the model: {exc}")
        st.stop()
    st.caption(
        f"Churn = Yes when model probability is at least {artifact['threshold']:.0%}. "
        "This model score is not a guarantee of what a customer will do."
    )
    st.sidebar.caption(f"Model: {artifact['model_id']}")
    mode = st.sidebar.radio("Mode", ["Single Prediction", "Batch Prediction"])
    if mode == "Single Prediction":
        single_prediction(artifact)
    else:
        batch_prediction(artifact)


if __name__ == "__main__":
    main()
