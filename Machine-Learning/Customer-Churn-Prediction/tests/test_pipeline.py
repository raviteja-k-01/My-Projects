"""Validate raw data, independent evaluation, artifact reuse, and explanations."""

from hashlib import sha256
from io import StringIO
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.preprocess import (
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    build_pipeline,
    load_artifact,
    load_dataset,
    predict_batch,
    predict_probabilities,
    prepare_features,
    prepare_training_data,
    read_customer_csv,
)
from src.train import calculate_metrics, split_dataset, train

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "telecom.csv"


def test_actual_dataset_audit_and_preservation():
    before = sha256(RAW_DATA.read_bytes()).hexdigest()
    original = pd.read_csv(RAW_DATA)
    customers, profile = load_dataset(RAW_DATA)
    assert original.shape == (7042, 21)
    assert original["Churn"].value_counts().to_dict() == {"Yes": 3521, "No": 3521}
    assert original.replace(r"^\s*$", np.nan, regex=True).isna().sum().sum() == 0
    assert len(customers) == 4053
    assert customers["customerID"].is_unique
    assert customers["Churn"].value_counts().to_dict() == {"No": 2518, "Yes": 1535}
    pd.testing.assert_frame_equal(customers, original.drop_duplicates())
    assert isinstance(profile, dict) and profile
    assert sha256(RAW_DATA.read_bytes()).hexdigest() == before


def test_split_has_disjoint_customers_and_reproducible_membership():
    customers, _ = load_dataset(RAW_DATA)
    x_train, x_test, y_train, y_test = split_dataset(customers)
    assert (len(x_train), len(x_test)) == (3242, 811)
    assert y_train.value_counts().to_dict() == {0: 2014, 1: 1228}
    assert y_test.value_counts().to_dict() == {0: 504, 1: 307}
    assert set(x_train.index).isdisjoint(x_test.index)
    train_ids = set(customers.loc[x_train.index, "customerID"])
    test_ids = set(customers.loc[x_test.index, "customerID"])
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == set(customers["customerID"])
    for first, second in zip((x_train, x_test, y_train, y_test), split_dataset(customers)):
        assert first.equals(second)
    with pytest.raises(ValueError, match="customerID"):
        split_dataset(pd.read_csv(RAW_DATA))


def test_duplicate_records_removed_but_conflicting_ids_rejected(raw_customers, tmp_path):
    path = tmp_path / "customers.csv"
    duplicated = pd.concat([raw_customers.iloc[:10], raw_customers.iloc[:2]], ignore_index=True)
    duplicated.to_csv(path, index=False)
    customers, _ = load_dataset(path)
    assert len(customers) == 10
    assert customers.index.tolist() == list(range(10))
    duplicated.loc[10, "Churn"] = "No"
    duplicated.to_csv(path, index=False)
    with pytest.raises(ValueError, match="(?i)customer|conflict"):
        load_dataset(path)


@pytest.mark.parametrize("bad_id", [None, "", " "])
def test_missing_customer_identity_rejected(raw_customers, tmp_path, bad_id):
    raw_customers.loc[0, "customerID"] = bad_id
    path = tmp_path / "customers.csv"
    raw_customers.to_csv(path, index=False)
    with pytest.raises(ValueError, match="(?i)customer"):
        load_dataset(path)


def test_raw_schema_target_encoding_and_column_order(raw_customers):
    x, y = prepare_training_data(raw_customers)
    assert list(x.columns) == list(FEATURE_COLUMNS)
    assert x.shape == (80, 19)
    assert "customerID" not in x and "Churn" not in x
    np.testing.assert_array_equal(y, raw_customers["Churn"].eq("Yes").astype(int))
    assert pd.isna(x.loc[4, "TotalCharges"])
    reordered = prepare_features(raw_customers.iloc[:, ::-1])
    pd.testing.assert_frame_equal(x, reordered)


@pytest.mark.parametrize("target", ["Maybe", "1", None, ""])
def test_invalid_target_rejected(raw_customers, target):
    raw_customers.loc[0, "Churn"] = target
    with pytest.raises(ValueError, match="Churn"):
        prepare_training_data(raw_customers)


def test_required_columns_and_duplicate_headers_rejected(raw_customers):
    with pytest.raises(ValueError, match="PaymentMethod"):
        prepare_features(raw_customers.drop(columns="PaymentMethod"))
    with pytest.raises(ValueError, match="Churn"):
        prepare_training_data(raw_customers.drop(columns="Churn"))
    duplicated = pd.concat([raw_customers, raw_customers[["gender"]]], axis=1)
    with pytest.raises(ValueError, match="(?i)duplicate"):
        prepare_features(duplicated)
    with pytest.raises(ValueError, match="(?i)duplicate"):
        read_customer_csv(StringIO('"gender","gender"\nFemale,Male\n'))


@pytest.mark.parametrize("column,value", [
    ("MonthlyCharges", "not a number"),
    ("MonthlyCharges", np.inf),
    ("TotalCharges", -1),
    ("tenure", -1),
    ("tenure", 2.5),
    ("SeniorCitizen", 2),
    ("SeniorCitizen", 0.5),
    ("PaymentMethod", "Bitcoin"),
    ("InternetService", "DSL typo"),
])
def test_invalid_feature_values_rejected(raw_customers, column, value):
    raw_customers[column] = raw_customers[column].astype(object)
    raw_customers.loc[0, column] = value
    with pytest.raises(ValueError, match=column):
        prepare_features(raw_customers)


@pytest.mark.parametrize("changes,error_column", [
    ({"PhoneService": "No", "MultipleLines": "Yes"}, "MultipleLines"),
    ({"PhoneService": "No", "MultipleLines": "No"}, "MultipleLines"),
    ({"PhoneService": "Yes", "MultipleLines": "No phone service"}, "MultipleLines"),
    ({"InternetService": "No"}, "OnlineSecurity"),
    ({"InternetService": "DSL", "OnlineSecurity": "No internet service"}, "OnlineSecurity"),
    ({"InternetService": "Fiber optic", "TechSupport": "No internet service"}, "TechSupport"),
    ({"InternetService": "DSL", "StreamingTV": "No internet service"}, "StreamingTV"),
    ({"InternetService": "Fiber optic", "StreamingMovies": "No internet service"}, "StreamingMovies"),
])
def test_inconsistent_service_information_rejected(raw_customers, changes, error_column):
    row = raw_customers.iloc[[0]].copy()
    for column, value in changes.items():
        row[column] = value
    with pytest.raises(ValueError, match=error_column):
        prepare_features(row)


def test_absent_services_and_missing_relationship_fields_are_valid(raw_customers, small_artifact):
    row = raw_customers.iloc[[0]].copy()
    row["PhoneService"] = "No"
    row["MultipleLines"] = "No phone service"
    row["InternetService"] = "No"
    for column in ("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                   "StreamingTV", "StreamingMovies"):
        row[column] = "No internet service"
    assert np.isfinite(predict_probabilities(small_artifact, row)).all()
    row["PhoneService"] = None
    row["InternetService"] = None
    assert np.isfinite(predict_probabilities(small_artifact, row)).all()


@pytest.mark.parametrize("content,error", [
    ("", "empty"),
    ("gender,tenure\nFemale,12,extra\n", "row 2"),
    ("gender,tenure\nFemale\n", "row 2"),
    ('gender,tenure\n"Female,12\n', "Malformed"),
])
def test_malformed_csv_reports_understandable_error(content, error):
    with pytest.raises(ValueError, match=error):
        read_customer_csv(StringIO(content))


def test_missing_cells_are_imputed_by_fitted_pipeline(raw_customers, small_artifact):
    rows = raw_customers.iloc[60:63].copy()
    rows.loc[rows.index[0], "TotalCharges"] = " "
    rows.loc[rows.index[1], "PaymentMethod"] = None
    rows.loc[rows.index[2], "MonthlyCharges"] = np.nan
    probabilities = predict_probabilities(small_artifact, rows)
    assert probabilities.shape == (3,)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_imputer_and_encoder_fit_only_training_data(raw_customers):
    train_rows = raw_customers.iloc[:60].copy()
    test_rows = raw_customers.iloc[60:].copy()
    train_rows["PaymentMethod"] = "Electronic check"
    test_rows["PaymentMethod"] = "Mailed check"
    test_rows["TotalCharges"] = 999999.0
    _, target = prepare_training_data(train_rows)
    pipe = build_pipeline().set_params(classifier__n_estimators=8)
    pipe.fit(train_rows, target)
    preprocessor = pipe.named_steps["preprocess"]
    numeric = preprocessor.named_transformers_["numeric"].named_steps["imputer"]
    categorical = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
    expected = prepare_features(train_rows)[NUMERIC_COLUMNS].median().to_numpy()
    np.testing.assert_allclose(numeric.statistics_, expected)
    assert all("Mailed check" not in category for category in categorical.categories_)
    before_statistics = numeric.statistics_.copy()
    before_categories = [values.copy() for values in categorical.categories_]
    assert np.isfinite(pipe.predict_proba(test_rows)).all()
    np.testing.assert_array_equal(numeric.statistics_, before_statistics)
    for before, after in zip(before_categories, categorical.categories_):
        np.testing.assert_array_equal(before, after)
    assert set(preprocessor.feature_names_in_) == set(FEATURE_COLUMNS)


def test_metrics_use_yes_as_positive_and_fixed_threshold():
    metrics = calculate_metrics(np.array([0, 0, 0, 1, 1, 1]),
                                np.array([0.1, 0.6, 0.2, 0.8, 0.4, 0.7]))
    assert metrics["confusion_matrix_no_yes"] == [[2, 1], [1, 2]]
    assert metrics["roc_auc"] == pytest.approx(8 / 9)
    for name in ("accuracy", "precision_churn", "recall_churn", "f1_churn"):
        assert metrics[name] == pytest.approx(2 / 3)
    at_boundary = calculate_metrics(np.array([0, 1]), np.array([0.49, 0.5]), threshold=0.5)
    assert at_boundary["confusion_matrix_no_yes"] == [[1, 0], [0, 1]]


def test_serialized_pipeline_preserves_probabilities_exactly(raw_customers, small_artifact, tmp_path):
    held_out = raw_customers.iloc[60:]
    before = predict_probabilities(small_artifact, held_out)
    path = tmp_path / "pipeline.joblib"
    joblib.dump(small_artifact, path)
    restored = load_artifact(path)
    np.testing.assert_array_equal(before, predict_probabilities(restored, held_out))
    np.testing.assert_array_equal(before, restored["pipeline"].predict_proba(held_out)[:, 1])
    np.testing.assert_array_equal(before, predict_probabilities(restored, held_out.iloc[:, ::-1]))


def test_single_and_batch_predictions_agree(raw_customers, small_artifact):
    rows = raw_customers.iloc[60:65].copy()
    batch = predict_batch(small_artifact, rows)
    expected = np.array([predict_probabilities(small_artifact, rows.iloc[[i]])[0]
                         for i in range(len(rows))])
    np.testing.assert_array_equal(batch["Churn_Probability"].to_numpy(), expected)
    assert batch["Predicted_Churn"].tolist() == np.where(expected >= 0.5, "Yes", "No").tolist()
    assert batch["Validation_Error"].fillna("").eq("").all()
    assert batch["customerID"].tolist() == rows["customerID"].tolist()


def test_batch_retains_valid_rows_and_flags_invalid_rows(raw_customers, small_artifact):
    rows = raw_customers.iloc[60:63].copy()
    rows.index = [7, 7, 9]
    rows.iloc[1, rows.columns.get_loc("MonthlyCharges")] = -10
    original = rows.copy(deep=True)
    result = predict_batch(small_artifact, rows)
    assert len(result) == 3
    assert result.index.tolist() == rows.index.tolist()
    assert result.iloc[[0, 2]]["Churn_Probability"].notna().all()
    assert pd.isna(result.iloc[1]["Churn_Probability"])
    assert "MonthlyCharges" in result.iloc[1]["Validation_Error"]
    assert result.iloc[[0, 2]]["Validation_Error"].fillna("").eq("").all()
    pd.testing.assert_frame_equal(rows, original)
    with pytest.raises(ValueError, match="PaymentMethod"):
        predict_batch(small_artifact, rows.drop(columns="PaymentMethod"))


def test_all_invalid_batch_records_remain_visible(raw_customers, small_artifact):
    rows = raw_customers.iloc[:3].copy()
    rows["MonthlyCharges"] = -1
    result = predict_batch(small_artifact, rows)
    assert len(result) == 3
    assert result["Churn_Probability"].isna().all()
    assert result["Validation_Error"].str.contains("MonthlyCharges").all()


def test_shap_explains_transformed_inputs_and_reconstructs_probability(raw_customers, small_artifact):
    from src.explain import explain_predictions

    raw = raw_customers.iloc[60:63]
    explanation = explain_predictions(small_artifact, raw)
    pipe = small_artifact["pipeline"]
    transformed = pipe.named_steps["preprocess"].transform(prepare_features(raw))
    np.testing.assert_allclose(explanation.data, transformed)
    assert list(explanation.feature_names) == list(pipe.named_steps["preprocess"].get_feature_names_out())
    assert explanation.values.shape == transformed.shape
    margin = pipe.named_steps["classifier"].predict(transformed, output_margin=True)
    explained_margin = explanation.base_values + explanation.values.sum(axis=1)
    np.testing.assert_allclose(explained_margin, margin, atol=1e-5)
    np.testing.assert_allclose(1 / (1 + np.exp(-explained_margin)),
                               predict_probabilities(small_artifact, raw), atol=1e-6)


def test_full_training_reproduces_saved_evaluation(tmp_path):
    report = train(raw_data=RAW_DATA, output_dir=tmp_path)
    artifact = load_artifact(tmp_path / "model" / "churn_pipeline.joblib")
    written = json.loads(
        (tmp_path / "reports" / "metrics.json").read_text(encoding="utf-8"),
        parse_constant=lambda value: pytest.fail(f"Report contains non-JSON value: {value}"),
    )
    assert written == report
    assert (tmp_path / "reports" / "split.csv").is_file()
    assert (tmp_path / "reports" / "test_predictions.csv").is_file()
    customers, _ = load_dataset(RAW_DATA)
    _, x_test, _, y_test = split_dataset(customers)
    probabilities = predict_probabilities(artifact, x_test)
    independent = calculate_metrics(y_test, probabilities, threshold=artifact["threshold"])
    committed = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
    for key, value in independent.items():
        assert report[key] == value
        assert committed[key] == value
    for name in ("preprocess.py", "train.py"):
        expected_hash = sha256((ROOT / "src" / name).read_bytes()).hexdigest()
        assert report["source_sha256"][name] == expected_hash
        assert committed["source_sha256"][name] == expected_hash
    split = pd.read_csv(tmp_path / "reports" / "split.csv")
    assert split["customerID"].is_unique
    assert split["partition"].value_counts().to_dict() == {"train": 3242, "test": 811}
    assert set(split.loc[split["partition"].eq("test"), "source_row"]) == set(x_test.index)
    saved_predictions = pd.read_csv(tmp_path / "reports" / "test_predictions.csv")
    np.testing.assert_array_equal(saved_predictions["actual_churn"], y_test)
    np.testing.assert_allclose(saved_predictions["Churn_Probability"], probabilities, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(saved_predictions["Predicted_Churn"], probabilities >= artifact["threshold"])
    canonical_artifact = load_artifact(ROOT / "model" / "churn_pipeline.joblib")
    np.testing.assert_array_equal(probabilities, predict_probabilities(canonical_artifact, x_test))
