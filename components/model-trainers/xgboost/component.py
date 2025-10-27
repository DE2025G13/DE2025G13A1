import argparse
import pandas as pd
import xgboost as xgb
import joblib
import os

def train_model(training_data_path: str, model_artifact_path: str):
    print("Starting XGBoost training process.")
    X_train = pd.read_csv(os.path.join(training_data_path, "x_train.csv"))
    y_train = pd.read_csv(os.path.join(training_data_path, "y_train.csv"))

    y_min = int(y_train.iloc[:, 0].min())
    y_max = int(y_train.iloc[:, 0].max())
    y_train_mapped = y_train.iloc[:, 0] - y_min
    
    print(f"Mapping quality scores: [{y_min}, {y_max}] -> [0, {y_max - y_min}]")
    
    model = xgb.XGBClassifier(
        objective="multi:softmax",
        eval_metric="mlogloss",
        n_estimators=150,
        random_state=42
    )
    
    print("Fitting the XGBoost model to the training data.")
    model.fit(X_train, y_train_mapped)
    print("XGBoost model fitting has completed.")
    
    os.makedirs(os.path.dirname(model_artifact_path), exist_ok=True)
    
    model_package = {
        "model": model,
        "quality_offset": y_min,
        "model_type": "xgboost"
    }
    
    joblib.dump(model_package, model_artifact_path)
    print(f"XGBoost model saved with quality offset {y_min} to {model_artifact_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data_path", type=str, required=True)
    parser.add_argument("--model_artifact_path", type=str, required=True)
    args = parser.parse_args()
    train_model(args.training_data_path, args.model_artifact_path)