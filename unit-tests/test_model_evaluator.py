import pandas as pd
import os
import pytest
import joblib
import importlib.util
from sklearn.ensemble import RandomForestClassifier
import json

EVALUATOR_PATH = "components/model-evaluator/component.py"

def import_evaluate():
    spec = importlib.util.spec_from_file_location("evaluator", EVALUATOR_PATH)
    if not spec:
        pytest.skip(f"Module not found: {EVALUATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'evaluate_and_decide'):
        pytest.fail(f"Function 'evaluate_and_decide' not found")
    return module.evaluate_and_decide

evaluate_and_decide = import_evaluate()

@pytest.fixture
def mock_evaluation_data(tmp_path):
    X_train = pd.DataFrame({'f1': list(range(20)), 'f2': list(range(20, 40))})
    y_train = pd.DataFrame({'quality': [5]*10 + [6]*10})
    X_test = pd.DataFrame({'f1': list(range(5)), 'f2': list(range(5, 10))})
    y_test = pd.DataFrame({'quality': [5, 5, 6, 6, 6]})
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    test_dir.mkdir()
    X_train.to_csv(train_dir / "x_train.csv", index=False)
    y_train.to_csv(train_dir / "y_train.csv", index=False)
    X_test.to_csv(test_dir / "x_test.csv", index=False)
    y_test.to_csv(test_dir / "y_test.csv", index=False)
    rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
    rf_model.fit(X_train, y_train.values.ravel())
    xgb_model = RandomForestClassifier(n_estimators=10, random_state=42)
    xgb_model.fit(X_train, y_train.values.ravel())
    svm_model = RandomForestClassifier(n_estimators=10, random_state=42)
    svm_model.fit(X_train, y_train.values.ravel())
    rf_dir = tmp_path / "rf"
    xgb_dir = tmp_path / "xgb"
    svm_dir = tmp_path / "svm"
    rf_dir.mkdir()
    xgb_dir.mkdir()
    svm_dir.mkdir()
    joblib.dump(rf_model, rf_dir / "model.joblib")
    joblib.dump(xgb_model, xgb_dir / "model.joblib")
    joblib.dump(svm_model, svm_dir / "model.joblib")
    return str(train_dir), str(test_dir), str(rf_dir), str(xgb_dir), str(svm_dir)

def test_evaluator_runs(mock_evaluation_data, tmp_path):
    train_dir, test_dir, rf_dir, xgb_dir, svm_dir = mock_evaluation_data
    decision_path = tmp_path / "decision.txt"
    uri_path = tmp_path / "uri.txt"
    metrics_path = tmp_path / "metrics.json"
    try:
        evaluate_and_decide(
            train_dir, test_dir, rf_dir, xgb_dir, svm_dir,
            "test-bucket", "test.joblib", "config-bucket", "config.json",
            str(decision_path), str(uri_path), str(metrics_path)
        )
    except Exception as e:
        if "credentials" in str(e).lower() or "authentication" in str(e).lower():
            pytest.skip("Skipping due to GCS authentication requirement")
        raise
    assert decision_path.exists(), "Decision file not created"
    assert uri_path.exists(), "URI file not created"
    assert metrics_path.exists(), "Metrics file not created"
    with open(metrics_path) as f:
        metrics = json.load(f)
    assert "decision" in metrics, "Metrics missing decision"
    assert "selected_model" in metrics, "Metrics missing selected_model"
    assert "all_candidates" in metrics, "Metrics missing all_candidates"