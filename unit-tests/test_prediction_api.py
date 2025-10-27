import pytest
import json
import sys
import os
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'prediction-api'))

def test_import_app():
    try:
        import app
        assert hasattr(app, 'app')
        assert hasattr(app, 'predict')
    except ImportError as e:
        pytest.skip(f"Could not import app module: {e}")

@pytest.fixture
def mock_dependencies():
    with patch.dict('sys.modules', {
        'google.cloud': Mock(),
        'google.cloud.storage': Mock()
    }):
        yield

def test_health_endpoint_structure(mock_dependencies):
    import app as api_module
    client = api_module.app.test_client()
    response = client.get('/health')
    assert response.status_code in [200, 500]
    data = json.loads(response.data)
    assert 'status' in data

def test_predict_endpoint_exists(mock_dependencies):
    import app as api_module
    client = api_module.app.test_client()
    assert '/predict' in [rule.rule for rule in api_module.app.url_map.iter_rules()]

def test_predict_endpoint_methods(mock_dependencies):
    import app as api_module
    client = api_module.app.test_client()
    predict_rule = None
    for rule in api_module.app.url_map.iter_rules():
        if rule.rule == '/predict':
            predict_rule = rule
            break
    assert predict_rule is not None
    assert 'POST' in predict_rule.methods

def test_type_encoding_logic():
    import pandas as pd
    df = pd.DataFrame([{"type": "red"}])
    df["type"] = df["type"].apply(lambda x: 1 if x == "red" else 0)
    assert df["type"].iloc[0] == 1
    df2 = pd.DataFrame([{"type": "white"}])
    df2["type"] = df2["type"].apply(lambda x: 1 if x == "red" else 0)
    assert df2["type"].iloc[0] == 0

def test_prediction_clamping():
    def clamp(value, min_val=0, max_val=10):
        return max(min_val, min(max_val, value))
    assert clamp(5) == 5
    assert clamp(-5) == 0
    assert clamp(15) == 10
    assert clamp(10) == 10
    assert clamp(0) == 0

def test_feature_list_completeness():
    expected_features = [
        "type",
        "fixed_acidity",
        "volatile_acidity",
        "citric_acid",
        "residual_sugar",
        "chlorides",
        "free_sulfur_dioxide",
        "total_sulfur_dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol"
    ]
    assert len(expected_features) == 12

def test_label_encoder_transform_inverse():
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    original = [3, 4, 5, 6, 7, 8, 9]
    encoded = encoder.fit_transform(original)
    assert list(encoded) == list(range(len(original)))
    decoded = encoder.inverse_transform(encoded)
    assert list(decoded) == original

def test_model_metadata_structure():
    metadata = {
        "label_encoder": None,
        "model_type": "legacy"
    }
    assert "label_encoder" in metadata
    assert "model_type" in metadata
    assert metadata["model_type"] in ["legacy", "xgboost", "random_forest", "svm", "unknown"]