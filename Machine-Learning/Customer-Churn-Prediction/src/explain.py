# EXPLAINABILITY MODULE (SHAP + XGBOOST)

import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
from xgboost import XGBClassifier


# PATH SETUP

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, "model", "churn_model.pkl")
data_path = os.path.join(base_dir, "data", "telecom_preprocessed.csv")


# LOAD MODEL AND DATA

def load_model_for_explain():
    print("Loading trained XGBoost model...")
    model = joblib.load(model_path)
    print("Model loaded successfully.")
    return model

def load_data_for_explain():
    print("Loading preprocessed data for SHAP analysis...")
    df = pd.read_csv(data_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df


# GLOBAL EXPLANATION (Feature Importance)

def explain_global_importance(model, X):
    print("Generating global feature importance with SHAP...")
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("Global Feature Importance (SHAP)")
    plt.tight_layout()

    out_path = os.path.join(base_dir, "screenshots", "shap_global_importance.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    print(f"Global SHAP importance plot saved at: {out_path}")
    plt.close()

# LOCAL EXPLANATION (Single Prediction)

def explain_single_prediction(model, X, index=0):
    print(f"Generating SHAP explanation for instance #{index}...")
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[index], show=False)
    plt.title(f"Local SHAP Explanation (Row {index})")
    plt.tight_layout()

    out_path = os.path.join(base_dir, "screenshots", f"shap_local_explain_{index}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    print(f"Local SHAP explanation saved at: {out_path}")
    plt.close()


# MAIN EXECUTION (for standalone testing)

if __name__ == "__main__":
    model = load_model_for_explain()
    df = load_data_for_explain()

    # Separate features
    X = df.drop(columns=["Churn"])
    
    # Generate global importance
    explain_global_importance(model, X)

    # Generate single instance explanation
    explain_single_prediction(model, X, index=0)

    print(" SHAP explainability completed.")
