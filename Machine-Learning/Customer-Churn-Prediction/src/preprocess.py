
# STEP 2: DATA PREPROCESSING FOR CUSTOMER CHURN

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Path setup
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(base_dir, "data", "telecom.csv")

# Load dataset
df = pd.read_csv(file_path)
print("Data loaded for preprocessing. Shape:", df.shape)

# Convert TotalCharges to numeric (some may be blank strings)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing TotalCharges with median
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Drop customerID (irrelevant for prediction)
df.drop(columns=["customerID"], inplace=True)

# Encode categorical variables
cat_cols = df.select_dtypes(include=["object"]).columns
encoder = LabelEncoder()

for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

print(f"Encoded categorical columns: {list(cat_cols)}")

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
processed_df = pd.DataFrame(X_scaled, columns=X.columns)
processed_df["Churn"] = y.values

# SAVE PROCESSED DATA
processed_path = os.path.join(base_dir, "data", "telecom_preprocessed.csv")
processed_df.to_csv(processed_path, index=False)

print(f"Preprocessing complete! Saved processed data to: {processed_path}")
print("Final Shape:", processed_df.shape)
print("Preview:\n", processed_df.head())

