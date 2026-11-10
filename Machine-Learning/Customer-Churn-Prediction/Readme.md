
# 🧠 Customer Churn Prediction Using Machine Learning

### Predicting whether a telecom customer will stay or leave using an explainable XGBoost model

---

## 🚀 About the Project

This project predicts **customer churn** for a **telecom business**.  
The goal is to identify customers who are likely to discontinue their services,  
allowing the business to take proactive retention actions.

It’s a **complete end-to-end pipeline** — from raw data preprocessing and model training  
to an interactive **Streamlit web app** with **SHAP explainability**.

---

## 🧩 What’s Included

- ✅ Data cleaning & preprocessing  
- ✅ Feature encoding and missing value handling  
- ✅ Model training with **XGBoost**  
- ✅ Evaluation using accuracy, ROC-AUC, and F1-score  
- ✅ Deployment-ready **Streamlit app**  
- ✅ **SHAP** integration to explain predictions visually  

---

## 📊 Dataset Overview

The dataset (`telecom.csv`) contains **7,042 customer records** with 21 features such as:

- **Tenure** — number of months a customer has stayed  
- **Contract Type** — Month-to-month, One year, Two year  
- **InternetService**, **TechSupport**, **PaymentMethod**  
- **MonthlyCharges**, **TotalCharges**  
- **Churn** — target variable (Yes = churn, No = stay)

The data is based on the **Telco Customer Churn dataset** — a standard for churn prediction tasks.

---

## ⚙️ Feature Engineering & Preprocessing

- Handled missing values in `TotalCharges`
- Encoded categorical variables numerically
- Standardized features for model interpretability
- Split into training and testing sets (80/20)

---

## 🧠 Model Training (XGBoost)

Model trained using XGBoost — chosen for its strong performance on tabular data.

```python
XGBClassifier(
    n_estimators=250,
    learning_rate=0.05,
    max_depth=6,
    eval_metric='logloss',
    random_state=42
)
````

---

## 🧮 Model Evaluation

| Metric        | Score |
| :------------ | :---: |
| **Accuracy**  |  0.84 |
| **Precision** |  0.87 |
| **Recall**    |  0.88 |
| **F1-Score**  |  0.84 |
| **ROC-AUC**   |  0.92 |

📊 **Confusion Matrix**

```
[[560 145]
 [ 85 619]]
```

✅ 560 True Negatives
✅ 619 True Positives
⚠️ 145 False Positives
⚠️ 85 False Negatives

The model generalizes well without overfitting.

---

## 📊 Explainability with SHAP

Each prediction is accompanied by a **SHAP waterfall plot** that visualizes
how each feature contributed to the final decision (e.g., high monthly charges → churn risk).

---

## 🖥️ Streamlit Web App

Run the app locally:

```bash
streamlit run app/streamlit_app.py
```

### Features:

* 🔮 **Single Prediction:** Manually input customer details
* 📦 **Batch Prediction:** Upload CSV for mass inference
* 🔍 **Explain Prediction:** SHAP-based feature contribution plots

---

## 📁 Project Structure

```
Customer-Churn-Prediction/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── telecom.csv
│   └── telecom_preprocessed.csv
│
├── model/
│   └── churn_model.pkl
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── data_inspect.py
│   └── explain.py
│
├── screenshots/
│   ├── accuracy_report.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── shap_waterfall.png
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE

```

---

## 🧰 Tech Stack

* Python 3.10+
* Pandas, NumPy
* Scikit-learn
* XGBoost
* SHAP
* Streamlit
* Matplotlib, Seaborn

---

## 🧾 What I Learned

* Importance of **data preprocessing and encoding** in model quality
* How **feature importance** and **tenure** drive customer retention
* Building **explainable ML systems** with SHAP
* Streamlit makes deploying ML apps extremely quick and intuitive

---

## 🚧 Future Improvements

* Real-time inference API
* Automated retraining using live data
* Cloud deployment (AWS / Render / HuggingFace Spaces)

---

## 👤 Author

**Ravi Teja Kesagani**
📧 [raviteja.inboxx@gmail.com](mailto:raviteja.inboxx@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/ravitejakesagani1)
💻 [GitHub](https://github.com/raviteja-k-01)

---

