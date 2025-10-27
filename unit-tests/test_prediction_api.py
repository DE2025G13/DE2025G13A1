import pytest
import json
import joblib
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import importlib.util
import sys
import os

@pytest.fixture
def mock_model_no_encoder(tmp_path):
    X = pd.DataFrame({'f1': [1, 2, 3], 'f2': [4, 5, 6]})
    y = [5, 6, 7]
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    model_path = tmp_path / "model_no_encoder.joblib"
    joblib.dump(model, model_path)
    return str(model_path)

@pytest.fixture
def mock_model_with_encoder(tmp_path):
    X = pd.DataFrame({'f1': [1, 2, 3], 'f2': [4, 5, 6]})
    y = [5, 6, 7]
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    model.fit(X, y_encoded)
    model_package = {
        "model": model,
        "label_encoder": label_encoder,
        "model_type": "xgboost"
    }
    model_path = tmp_path / "model_with_encoder.joblib"
    joblib.dump(model_package, model_path)
    return str(model_path)

@pytest.fixture
def flask_app(mock_model_with_encoder, tmp_path):
    with patch('prediction-api.app.LOCAL_MODEL_PATH', str(mock_model_with_encoder)):
        with patch('prediction-api.app.download_model', return_value=True):
            spec = importlib.util.spec_from_file_location("api", "prediction-api/app.py")
            if not spec:
                pytest.skip("prediction-api/app.py not found")
            api_module = importlib.util.module_from_spec(spec)
            
            with patch.dict('sys.modules', {'google.cloud.storage': Mock()}):
                spec.loader.exec_module(api_module)
            
            api_module.LOCAL_MODEL_PATH = str(mock_model_with_encoder)
            api_module.load_model()
            
            app = api_module.app
            app.config['TESTING'] = True
            return app.test_client()

def test_health_endpoint_with_model(flask_app):
    response = flask_app.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'ok'
    assert data['model'] == 'loaded'

def test_predict_endpoint_valid_input(flask_app):
    input_data = {
        "type": "red",
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11,
        "total_sulfur_dioxide": 34,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    }
    response = flask_app.post('/predict', 
                               data=json.dumps(input_data),
                               content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'prediction' in data
    assert 0 <= data['prediction'] <= 10

def test_predict_endpoint_missing_field(flask_app):
    input_data = {
        "type": "red",
        "fixed_acidity": 7.4
    }
    response = flask_app.post('/predict',
                               data=json.dumps(input_data),
                               content_type='application/json')
    assert response.status_code == 400

def test_predict_endpoint_invalid_type(flask_app):
    input_data = {
        "type": "invalid",
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11,
        "total_sulfur_dioxide": 34,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    }
    response = flask_app.post('/predict',
                               data=json.dumps(input_data),
                               content_type='application/json')
    assert response.status_code in [200, 400]

@patch('prediction-api.app.joblib.load')
def test_model_loading_legacy_format(mock_load, tmp_path):
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    mock_load.return_value = model
    spec = importlib.util.spec_from_file_location("api", "prediction-api/app.py")
    api_module = importlib.util.module_from_spec(spec)
    with patch.dict('sys.modules', {'google.cloud.storage': Mock()}):
        spec.loader.exec_module(api_module)
    api_module.LOCAL_MODEL_PATH = "/tmp/test_model.joblib"
    with patch('os.path.exists', return_value=True):
        api_module.load_model()
    assert api_module.model is not None
    assert api_module.model_metadata['model_type'] == 'legacy'
    assert api_module.model_metadata['label_encoder'] is None

@patch('prediction-api.app.joblib.load')
def test_model_loading_with_encoder_format(mock_load, tmp_path):
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    label_encoder = LabelEncoder()
    label_encoder.fit([3, 4, 5, 6, 7])
    model_package = {
        "model": model,
        "label_encoder": label_encoder,
        "model_type": "xgboost"
    }
    mock_load.return_value = model_package
    spec = importlib.util.spec_from_file_location("api", "prediction-api/app.py")
    api_module = importlib.util.module_from_spec(spec)
    with patch.dict('sys.modules', {'google.cloud.storage': Mock()}):
        spec.loader.exec_module(api_module)
    api_module.LOCAL_MODEL_PATH = "/tmp/test_model.joblib"
    with patch('os.path.exists', return_value=True):
        api_module.load_model()
    assert api_module.model is not None
    assert api_module.model_metadata['model_type'] == 'xgboost'
    assert api_module.model_metadata['label_encoder'] is not None