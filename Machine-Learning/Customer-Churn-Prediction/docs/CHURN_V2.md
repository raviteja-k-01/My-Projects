# Customer Churn v2 — safe correction candidate

**Status: separate candidate, not a validated replacement of the existing project.**

The existing `src/preprocess.py`, `src/train.py`, `app/streamlit_app.py`, `model/churn_model.json`, and original report files are unchanged.

## Why this revision exists

- Old scaling and categorical encoding were fitted before train/test splitting.
- The old Streamlit interface used inconsistent categorical codes and sent unscaled features to a scaled-input model.
- The existing README uses precision and recall without identifying the class; reported values are not mutually consistent with its confusion matrix if both refer to churn.

## Train and evaluate

From the `Customer-Churn-Prediction` directory, with the project's requirements installed:

```bash
python -m src.train_v2
```

This reads **raw** `data/telecom.csv`, splits 80/20 with stratification, fits the imputer/scaler/one-hot encoder on training data only, and then trains XGBoost. The positive label is **Churn = Yes**, and the classification threshold is **0.50** for both evaluation and the app.

New (not old) artifacts are created:

- `model/churn_pipeline_v2.joblib` — XGBoost **and fitted preprocessing**, stored together.
- `reports/churn_metrics_v2.json` — test-set metrics, class names, confusion matrix, train/test counts.

Never copy previously reported `0.92 ROC-AUC, 0.87 precision, 0.88 recall` to a resume without matching, reproducible evidence from this revised pipeline. Do not overwrite old screenshots or model artifacts while evaluating the candidate.

## New app

```bash
streamlit run app/streamlit_app_v2.py
```

The v2 app expects **raw**, human-readable categories and numerical charges, not the old preprocessed CSV. Both single-row and batch inference reuse the fitted pipeline. SHAP is computed after applying the same pipeline transformation. Batch CSVs require all 19 input columns; extra fields are ignored for prediction.

## Local smoke tests

```bash
python -m pytest tests/test_churn_v2.py -q
```

**Release gate:** Train on the full raw dataset, inspect `reports/churn_metrics_v2.json`, manually test single/batch Streamlit predictions and SHAP, and only then merge or replace the old application. GitHub source review alone does not validate the original accuracy claims.
