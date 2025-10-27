import pytest
import json
import sys
import os
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'prediction-ui'))

def test_import_app():
    try:
        with patch.dict('sys.modules', {
            'google.auth': Mock(),
            'google.auth.transport.requests': Mock(),
            'google.oauth2.id_token': Mock()
        }):
            import app
            assert hasattr(app, 'app')
            assert hasattr(app, 'get_quality_rating')
    except ImportError as e:
        pytest.skip(f"Could not import app module: {e}")

@pytest.fixture
def mock_dependencies():
    with patch.dict('sys.modules', {
        'google.auth': Mock(),
        'google.auth.transport.requests': Mock(),
        'google.oauth2.id_token': Mock()
    }):
        yield

def test_get_quality_rating_poor(mock_dependencies):
    import app as ui_module
    rating = ui_module.get_quality_rating(3)
    assert rating['stars'] == 0
    assert rating['label'] == 'Poor Quality'
    assert 0 <= rating['percentage'] <= 100
    assert isinstance(rating['stars'], int)
    assert isinstance(rating['label'], str)
    assert isinstance(rating['percentage'], (int, float))

def test_get_quality_rating_below_average(mock_dependencies):
    import app as ui_module
    rating = ui_module.get_quality_rating(5)
    assert rating['stars'] == 1
    assert rating['label'] == 'Below Average'
    assert 0 <= rating['percentage'] <= 100

def test_get_quality_rating_average(mock_dependencies):
    import app as ui_module
    rating = ui_module.get_quality_rating(6)
    assert rating['stars'] == 2
    assert rating['label'] == 'Average Quality'
    assert 0 <= rating['percentage'] <= 100

def test_get_quality_rating_good(mock_dependencies):
    import app as ui_module
    rating = ui_module.get_quality_rating(8)
    assert rating['stars'] == 3
    assert rating['label'] == 'Good Quality'
    assert 0 <= rating['percentage'] <= 100

def test_get_quality_rating_excellent(mock_dependencies):
    import app as ui_module
    rating = ui_module.get_quality_rating(9)
    assert rating['stars'] == 4
    assert rating['label'] == 'Excellent Quality'
    assert 0 <= rating['percentage'] <= 100

def test_quality_rating_boundaries(mock_dependencies):
    import app as ui_module
    rating_min = ui_module.get_quality_rating(0)
    assert rating_min['stars'] >= 0
    assert rating_min['percentage'] == 0
    rating_max = ui_module.get_quality_rating(10)
    assert rating_max['stars'] >= 0
    assert rating_max['percentage'] == 100

def test_quality_rating_clamping(mock_dependencies):
    import app as ui_module
    rating_low = ui_module.get_quality_rating(-5)
    assert 0 <= rating_low['percentage'] <= 100
    assert rating_low['stars'] >= 0
    rating_high = ui_module.get_quality_rating(15)
    assert 0 <= rating_high['percentage'] <= 100
    assert rating_high['stars'] >= 0

def test_index_page_exists(mock_dependencies):
    import app as ui_module
    client = ui_module.app.test_client()
    response = client.get('/')
    assert response.status_code == 200
    assert b'Wine Quality' in response.data or b'wine' in response.data.lower()

def test_health_endpoint(mock_dependencies):
    import app as ui_module
    client = ui_module.app.test_client()
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'

def test_predict_endpoint_exists(mock_dependencies):
    import app as ui_module
    assert '/predict' in [rule.rule for rule in ui_module.app.url_map.iter_rules()]

def test_predict_endpoint_methods(mock_dependencies):
    import app as ui_module
    predict_rule = None
    for rule in ui_module.app.url_map.iter_rules():
        if rule.rule == '/predict':
            predict_rule = rule
            break
    assert predict_rule is not None
    assert 'POST' in predict_rule.methods

def test_required_features_list(mock_dependencies):
    import app as ui_module
    expected_features = [
        "type", "fixed_acidity", "volatile_acidity", "citric_acid",
        "residual_sugar", "chlorides", "free_sulfur_dioxide",
        "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"
    ]
    assert ui_module.REQUIRED_FEATURES == expected_features

def test_min_max_quality_constants(mock_dependencies):
    import app as ui_module
    assert ui_module.MIN_QUALITY_SCORE == 0
    assert ui_module.MAX_QUALITY_SCORE == 10
    assert ui_module.MIN_QUALITY_SCORE < ui_module.MAX_QUALITY_SCORE

def test_quality_rating_all_scores(mock_dependencies):
    import app as ui_module
    for score in range(0, 11):
        rating = ui_module.get_quality_rating(score)
        assert 'stars' in rating
        assert 'label' in rating
        assert 'percentage' in rating
        assert 0 <= rating['stars'] <= 4
        assert 0 <= rating['percentage'] <= 100