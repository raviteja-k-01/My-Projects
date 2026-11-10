# ======================================================
# STEP 3: TRAIN XGBOOST MODEL FOR CUSTOMER CHURN
# ======================================================

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------
# PATH SETUP
# --------------------------------------
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(base_dir, "data", "telecom_preprocessed.csv")

# --------------------------------------
# LOAD DATA
# --------------------------------------
if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ Processed dataset not found at: {file_path}")

df = pd.read_csv(file_path)
print("✅ Loaded preprocessed dataset:", df.shape)

# --------------------------------------
# TRAIN-TEST SPLIT
# --------------------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Split complete: Train = {X_train.shape}, Test = {X_test.shape}")

# --------------------------------------
# MODEL INITIALIZATION
# --------------------------------------
model = XGBClassifier(
    n_estimators=250,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

# --------------------------------------
# TRAINING
# --------------------------------------
print("🚀 Training XGBoost model...")
model.fit(X_train, y_train)

# --------------------------------------
# EVALUATION
# --------------------------------------
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

acc = (y_pred == y_test).mean() * 100
roc_score = roc_auc_score(y_test, y_pred_proba)

print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_score:.4f}")
print(f"Accuracy: {acc:.2f}%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --------------------------------------
# CREATE / SAVE REPORTS DIRECTORY
# --------------------------------------
reports_dir = os.path.join(base_dir, "screenshots")
os.makedirs(reports_dir, exist_ok=True)

# ======================================================
# 📉 1️⃣ CONFUSION MATRIX
# ======================================================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
conf_path = os.path.join(reports_dir, "confusion_matrix.png")
plt.savefig(conf_path, bbox_inches='tight')
plt.close()
print(f"✅ Confusion matrix saved at: {conf_path}")

# ======================================================
# 📜 2️⃣ CLASSIFICATION REPORT (TEXT + CENTERED IMAGE)
# ======================================================
report = classification_report(y_test, y_pred, digits=2)
report_text = (
    f"Accuracy: {acc:.2f}%\n"
    f"ROC-AUC Score: {roc_score:.2f}\n\n"
    f"Classification Report:\n{report}"
)

# Save as text file
report_txt_path = os.path.join(reports_dir, "accuracy_report.txt")
with open(report_txt_path, "w") as f:
    f.write(report_text)
print(f"✅ Classification report text saved at: {report_txt_path}")

# Generate centered image version
width, height = 900, 500
img = Image.new("RGB", (width, height), color=(255, 255, 255))
draw = ImageDraw.Draw(img)

# Load font
try:
    font = ImageFont.truetype("arial.ttf", 18)
except:
    font = ImageFont.load_default()

# Center the text
text_bbox = draw.multiline_textbbox((0, 0), report_text, font=font)
text_width = text_bbox[2] - text_bbox[0]
text_height = text_bbox[3] - text_bbox[1]
x = (width - text_width) / 2
y = (height - text_height) / 2
draw.multiline_text((x, y), report_text, fill=(0, 0, 0), font=font, align="center")

acc_img_path = os.path.join(reports_dir, "accuracy_report.png")
img.save(acc_img_path)
print(f"✅ Centered accuracy report image saved at: {acc_img_path}")

# ======================================================
# 🌲 3️⃣ FEATURE IMPORTANCE
# ======================================================
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 5))
plt.barh(np.array(X.columns)[indices], importances[indices], color='green')
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("XGBoost Feature Importance")
plt.tight_layout()

fi_path = os.path.join(reports_dir, "feature_importance.png")
plt.savefig(fi_path, bbox_inches='tight')
plt.close()
print(f"✅ Feature importance saved at: {fi_path}")

# ======================================================
# SAVE MODEL IN MODERN XGBOOST JSON FORMAT (Recommended)
# ======================================================
model_dir = os.path.join(base_dir, "model")
os.makedirs(model_dir, exist_ok=True)

# Save model in JSON format for compatibility with newer XGBoost versions
json_model_path = os.path.join(model_dir, "churn_model.json")
model.save_model(json_model_path)

print(f"\n✅ Model saved successfully in JSON format at: {json_model_path}")
print("🎯 Training and reporting completed successfully!")
