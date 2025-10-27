import pandas as pd
import os
import pytest
import importlib.util

SPLITTER_PATH = "components/train-test-splitter/component.py"

def import_split_data():
    spec = importlib.util.spec_from_file_location("splitter", SPLITTER_PATH)
    if not spec:
        pytest.skip(f"Module not found: {SPLITTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'split_data'):
        pytest.fail(f"Function 'split_data' not found in {SPLITTER_PATH}")
    return module.split_data

split_data = import_split_data()

@pytest.fixture
def mock_data():
    data = {
        'type': [0, 1] * 50,
        'fixed_acidity': [7.0 + i*0.1 for i in range(100)],
        'volatile_acidity': [0.5 + i*0.01 for i in range(100)],
        'quality': [5]*60 + [6]*30 + [7]*10
    }
    return pd.DataFrame(data)

def test_train_test_split(mock_data, tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    mock_data.to_csv(input_dir / "wine.csv", index=False)
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    split_data(str(input_dir), str(train_dir), str(test_dir))
    assert (train_dir / "x_train.csv").exists(), "x_train.csv not created"
    assert (train_dir / "y_train.csv").exists(), "y_train.csv not created"
    assert (test_dir / "x_test.csv").exists(), "x_test.csv not created"
    assert (test_dir / "y_test.csv").exists(), "y_test.csv not created"
    X_train = pd.read_csv(train_dir / "x_train.csv")
    y_train = pd.read_csv(train_dir / "y_train.csv")
    X_test = pd.read_csv(test_dir / "x_test.csv")
    y_test = pd.read_csv(test_dir / "y_test.csv")
    assert len(X_train) + len(X_test) == len(mock_data), "Split sizes don't match"
    assert "quality" not in X_train.columns, "Target should not be in features"
    assert "quality" in y_train.columns, "Target should be in labels"
    assert len(X_train) > len(X_test), "Training set should be larger"