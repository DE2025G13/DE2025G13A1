import pandas as pd
import os
import pytest
import importlib.util

INGESTION_PATH = "components/data-ingestion/component.py"

def import_load_data():
    spec = importlib.util.spec_from_file_location("data_ingestion", INGESTION_PATH)
    if not spec:
        pytest.skip(f"Module not found: {INGESTION_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'load_data'):
        pytest.fail(f"Function 'load_data' not found in {INGESTION_PATH}")
    return module.load_data

load_data = import_load_data()

@pytest.fixture
def mock_raw_data():
    return pd.DataFrame({
        "Id": [1, 2, 3, 4],
        "type": ["red", "white", "red", "white"],
        "fixed_acidity": [7.4, 6.8, 7.2, 6.5],
        "volatile_acidity": [0.7, 0.6, 0.65, 0.58],
        "quality": [5, 6, 5, 7]
    })

def test_data_ingestion(mock_raw_data, tmp_path):
    input_path = tmp_path / "wine.csv"
    output_dir = tmp_path / "output"
    mock_raw_data.to_csv(input_path, index=False)
    load_data(str(input_path), str(output_dir))
    output_file = output_dir / "wine.csv"
    assert output_file.exists(), "Output file not created"
    df = pd.read_csv(output_file)
    assert "Id" not in df.columns, "Id column should be dropped"
    assert df["type"].dtype in [int, 'int64'], "Type should be encoded as integer"
    assert set(df["type"].unique()).issubset({0, 1}), "Type should be 0 or 1"
    assert len(df) == len(mock_raw_data), "Row count should match"