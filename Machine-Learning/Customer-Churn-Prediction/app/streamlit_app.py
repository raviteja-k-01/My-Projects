# ======================================================
# STREAMLIT APP : CUSTOMER CHURN PREDICTION (FINAL CLEAN VERSION)
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

# Must be the first Streamlit command
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# ======================================================
# LOAD TRAINED MODEL (CACHED)
# ======================================================
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, "model", "churn_model.json")

@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model(model_path)
    return model

model = load_model()

# ======================================================
# INITIALIZE SHAP EXPLAINER (CACHED)
# ======================================================
@st.cache_resource
def load_shap_explainer(_model):
    return shap.Explainer(_model)

explainer = load_shap_explainer(model)

# ======================================================
# FEATURE COLUMNS
# ======================================================
X_columns = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

# ======================================================
# PAGE LAYOUT
# ======================================================
st.title("💡 Customer Churn Prediction System")
st.sidebar.title("Select Mode")
mode = st.sidebar.radio("Mode:", ["🔮 Single Prediction", "📦 Batch Prediction"])

# ======================================================
# PREDICTION FUNCTION
# ======================================================
def make_prediction(data: pd.DataFrame):
    probs = model.predict_proba(data)[:, 1]
    preds = (probs >= 0.40).astype(int)
    return preds, probs

# ======================================================
# SINGLE PREDICTION
# ======================================================
if mode == "🔮 Single Prediction":
    st.subheader("Enter Customer Details")

    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (Months)", 0, 72, 12)
    phoneservice = st.selectbox("Phone Service", ["Yes", "No"])
    multiplelines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internetservice = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    onlinesecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    onlinebackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    deviceprotection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    techsupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streamingtv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streamingmovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperlessbilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    paymentmethod = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    monthlycharges = st.number_input("Monthly Charges", 0.0, 150.0, 70.0)
    totalcharges = st.number_input("Total Charges", 0.0, 10000.0, 2000.0)

    input_dict = {
        "gender": [1 if gender == "Male" else 0],
        "SeniorCitizen": [senior],
        "Partner": [1 if partner == "Yes" else 0],
        "Dependents": [1 if dependents == "Yes" else 0],
        "tenure": [tenure],
        "PhoneService": [1 if phoneservice == "Yes" else 0],
        "MultipleLines": [0 if multiplelines == "No" else (1 if multiplelines == "Yes" else 2)],
        "InternetService": [0 if internetservice == "DSL" else (1 if internetservice == "Fiber optic" else 2)],
        "OnlineSecurity": [0 if onlinesecurity == "No" else (1 if onlinesecurity == "Yes" else 2)],
        "OnlineBackup": [0 if onlinebackup == "No" else (1 if onlinebackup == "Yes" else 2)],
        "DeviceProtection": [0 if deviceprotection == "No" else (1 if deviceprotection == "Yes" else 2)],
        "TechSupport": [0 if techsupport == "No" else (1 if techsupport == "Yes" else 2)],
        "StreamingTV": [0 if streamingtv == "No" else (1 if streamingtv == "Yes" else 2)],
        "StreamingMovies": [0 if streamingmovies == "No" else (1 if streamingmovies == "Yes" else 2)],
        "Contract": [0 if contract == "Month-to-month" else (1 if contract == "One year" else 2)],
        "PaperlessBilling": [1 if paperlessbilling == "Yes" else 0],
        "PaymentMethod": [0 if paymentmethod == "Electronic check" else (1 if paymentmethod == "Mailed check" else (2 if paymentmethod == "Bank transfer (automatic)" else 3))],
        "MonthlyCharges": [monthlycharges],
        "TotalCharges": [totalcharges],
    }

    df_input = pd.DataFrame(input_dict)
    st.write("### Input Summary:", df_input)

    if st.button("Predict Churn"):
        pred, prob = make_prediction(df_input)
        churn = "CHURN" if pred[0] == 1 else "STAY"
        conf = round(prob[0]*100, 2) if pred[0] == 1 else round((1-prob[0])*100, 2)
        color = "red" if churn == "CHURN" else "green"
        st.markdown(
            f"<h3 style='color:{color};'>✅ Customer likely to {churn} (Confidence: {conf}%)</h3>",
            unsafe_allow_html=True,
        )
        st.session_state["last_input"] = df_input

    if "last_input" in st.session_state:
        st.subheader("🔍 Explain this prediction")
        if st.button("Show SHAP Explanation"):
            df_input = st.session_state["last_input"]
            shap.initjs()
            shap_values = explainer(df_input)
            st.subheader("📊 SHAP Feature Impact (Waterfall Plot)")
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)

# ======================================================
# BATCH PREDICTION
# ======================================================
elif mode == "📦 Batch Prediction":
    st.subheader("Upload a CSV file for batch predictions")
    uploaded = st.file_uploader("Choose CSV", type=["csv"])
    if uploaded:
        batch_df = pd.read_csv(uploaded)
        st.write("✅ File uploaded! Preview:")
        st.dataframe(batch_df.head())

        preds, probs = make_prediction(batch_df)
        batch_df["Predicted_Churn"] = ["Yes" if p == 1 else "No" for p in preds]
        batch_df["Confidence"] = np.round(np.where(preds == 1, probs, 1 - probs) * 100, 2)
        st.write("### Results:")
        st.dataframe(batch_df.head())

        csv = batch_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions", csv, "batch_predictions.csv", "text/csv")

if __name__ == "__main__":
    pass
