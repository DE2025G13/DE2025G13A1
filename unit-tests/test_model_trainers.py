import pandas as pd
import os
import pytest
import joblib
import importlib.util
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import SVC

TRAINER_PATHS = {
    "random-forest": "components/model-trainers/random-forest/component.py",
    "xgboost": "components/model-trainers/xgboost/component.py",
    "svm": "components/model-trainers/svm/component.py"
}

def import_train_model(trainer_name):
    path = TRAINER_PATHS[trainer_name]
    spec = importlib.util.spec_from_file_location(f"trainer_{trainer_name}", path)
    if not spec:
        pytest.skip(f"Module not found: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'train_model'):
        pytest.fail(f"Function 'train_model' not found in {path}")
    return module.train_model

@pytest.fixture
def mock_train_data(tmp_path):
    X_train = pd.DataFrame({
        'f1': [1, 2, 3, 4, 5, 6, 7, 8],
        'f2': [5, 6, 7, 8, 9, 10, 11, 12]
    })
    y_train = pd.DataFrame({'quality': [5, 6, 5, 6, 7, 7, 8, 8]})
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    X_train.to_csv(train_dir / "x_train.csv", index=False)
    y_train.to_csv(train_dir / "y_train.csv", index=False)
    return str(train_dir)

@pytest.mark.parametrize("trainer_name", TRAINER_PATHS.keys())
def test_model_trainer(trainer_name, mock_train_data, tmp_path):
    train_model_func = import_train_model(trainer_name)
    model_output_dir = tmp_path / f"{trainer_name}_model"
    train_model_func(mock_train_data, str(model_output_dir))
    model_file = model_output_dir / "model.joblib"
    assert model_file.exists(), f"Model artifact not found for {trainer_name}"
    assert model_file.stat().st_size > 100, f"Model file too small for {trainer_name}"
    loaded = joblib.load(model_file)
    if isinstance(loaded, dict) and "model" in loaded:
        model = loaded["model"]
        assert model is not None, f"Model object is None for {trainer_name}"
        if trainer_name == "xgboost":
            assert isinstance(model, xgb.XGBClassifier), f"Wrong model type for {trainer_name}"
            assert "label_encoder" in loaded, "XGBoost should have label_encoder"
        elif trainer_name == "random-forest":
            assert isinstance(model, RandomForestClassifier), f"Wrong model type for {trainer_name}"
    else:
        model = loaded
        assert model is not None, f"Model is None for {trainer_name}"