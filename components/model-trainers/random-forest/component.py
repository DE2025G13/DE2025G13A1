import argparse
import pandas as pd
import xgboost as xgb
import joblib
import os

def train_model(training_data_path: str, model_artifact_path: str):
    """Trains an XGBoost Classifier model."""
    print("--- Starting XGBoost Training ---")
    X_train = pd.read_csv(os.path.join(training_data_path, "x_train.csv"))
    y_train = pd.read_csv(os.path.join(training_data_path, "y_train.csv"))

    y_train_mapped = y_train.iloc[:, 0] - y_train.iloc[:, 0].min()
    
    model = xgb.XGBClassifier(
        objective='multi:softmax', 
        use_label_encoder=False, 
        eval_metric='mlogloss', 
        n_estimators=150,
        random_state=42
    )
    model.fit(X_train, y_train_mapped.values.ravel())
    
    os.makedirs(os.path.dirname(model_artifact_path), exist_ok=True)
    
    joblib.dump(model, model_artifact_path)
    print(f"XGBoost model saved to {model_artifact_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data_path', type=str, required=True)
    parser.add_argument('--model_artifact_path', type=str, required=True)
    args = parser.parse_args()
    train_model(args.training_data_path, args.model_artifact_path)