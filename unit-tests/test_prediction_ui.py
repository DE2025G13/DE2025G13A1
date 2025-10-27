import pytest
import importlib.util
from unittest.mock import Mock, patch, MagicMock
import json

@pytest.fixture
def flask_app():
    spec = importlib.util.spec_from_file_location("ui", "prediction-ui/app.py")
    if not spec:
        pytest.skip("prediction-ui/app.py not found")
    ui_module = importlib.util.module_from_spec(spec)
    with patch.dict('sys.modules', {'google.auth': Mock(), 'google.auth.transport.requests': Mock(), 'google.oauth2.id_token': Mock()}):
        spec.loader.exec_module(ui_module)
    app = ui_module.app
    app.config['TESTING'] = True
    return app.test_client(), ui_module

def test_index_page_loads(flask_app):
    client, _ = flask_app
    response = client.get('/')
    assert response.status_code == 200
    assert b'Wine Quality Predictor' in response.data

def test_health_endpoint(flask_app):
    client, _ = flask_app
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'

def test_get_quality_rating_function(flask_app):
    _, ui_module = flask_app
    rating = ui_module.get_quality_rating(3)
    assert rating['stars'] == 0
    assert rating['label'] == 'Poor Quality'
    rating = ui_module.get_quality_rating(6)
    assert rating['stars'] == 2
    assert rating['label'] == 'Average Quality'
    rating = ui_module.get_quality_rating(9)
    assert rating['stars'] == 4
    assert rating['label'] == 'Excellent Quality'

def test_quality_rating_boundaries(flask_app):
    _, ui_module = flask_app
    rating = ui_module.get_quality_rating(0)
    assert 0 <= rating['stars'] <= 4
    assert rating['percentage'] == 0
    rating = ui_module.get_quality_rating(10)
    assert 0 <= rating['stars'] <= 4
    assert rating['percentage'] == 100
    rating = ui_module.get_quality_rating(11)
    assert rating['percentage'] == 100

@patch('prediction-ui.app.get_identity_token')
@patch('prediction-ui.app.requests.post')
@patch('prediction-ui.app.PREDICTOR_API_URL', 'http://test-api')
def test_predict_success(mock_post, mock_token, flask_app):
    client, _ = flask_app
    mock_token.return_value = "test-token"
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"prediction": 6}
    mock_post.return_value = mock_response
    form_data = {
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
    response = client.post('/predict', data=form_data)
    assert response.status_code == 200
    assert b'Average Quality' in response.data

@patch('prediction-ui.app.PREDICTOR_API_URL', None)
def test_predict_no_api_url(flask_app):
    client, _ = flask_app
    form_data = {
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
    response = client.post('/predict', data=form_data)
    assert response.status_code == 200
    assert b'configuration error' in response.data.lower()

def test_predict_missing_field(flask_app):
    client, _ = flask_app
    form_data = {
        "type": "red",
        "fixed_acidity": 7.4
    }
    response = client.post('/predict', data=form_data)
    assert response.status_code == 200
    assert b'Missing required field' in response.data or b'error' in response.data.lower()