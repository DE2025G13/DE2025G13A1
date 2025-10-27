import argparse
import pandas as pd
import xgboost as xgb
import joblib
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

def train_model(training_data_path: str, model_artifact_path: str):
    print("Starting XGBoost training process.")
    X_train = pd.read_csv(os.path.join(training_data_path, "x_train.csv"))
    y_train = pd.read_csv(os.path.join(training_data_path, "y_train.csv"))
    
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train.iloc[:, 0])
    
    print(f"Original quality classes: {sorted(label_encoder.classes_)}")
    print(f"Encoded to: {list(range(len(label_encoder.classes_)))}")
    
    model = xgb.XGBClassifier(
        objective="multi:softmax",
        eval_metric="mlogloss",
        n_estimators=150,
        random_state=42
    )
    
    print("Fitting the XGBoost model to the training data.")
    model.fit(X_train, y_train_encoded)
    print("XGBoost model fitting has completed.")
    
    model_package = {
        "model": model,
        "label_encoder": label_encoder,
        "model_type": "xgboost"
    }
    
    os.makedirs(model_artifact_path, exist_ok=True)
    model_file_path = os.path.join(model_artifact_path, "model.joblib")
    joblib.dump(model_package, model_file_path)
    print(f"XGBoost model saved with label encoder to {model_file_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data_path", type=str, required=True)
    parser.add_argument("--model_artifact_path", type=str, required=True)
    args = parser.parse_args()
    train_model(args.training_data_path, args.model_artifact_path)