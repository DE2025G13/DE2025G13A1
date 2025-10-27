import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_model(training_data_path: str, model_artifact_path: str):
    print("Starting Random Forest training process.")
    X_train = pd.read_csv(os.path.join(training_data_path, "x_train.csv"))
    y_train = pd.read_csv(os.path.join(training_data_path, "y_train.csv"))
    model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=10)
    print("Fitting the Random Forest model to the training data.")
    model.fit(X_train, y_train.values.ravel())
    print("Model fitting has completed.")
    
    os.makedirs(model_artifact_path, exist_ok=True)
    model_file_path = os.path.join(model_artifact_path, "model.joblib")
    joblib.dump(model, model_file_path)
    print(f"Random Forest model has been saved to {model_file_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data_path", type=str, required=True, help="Path to the folder containing training data.")
    parser.add_argument("--model_artifact_path", type=str, required=True, help="Path to save the output model artifact.")
    args = parser.parse_args()
    try:
        train_model(args.training_data_path, args.model_artifact_path)
    except Exception as e:
        print("An error occurred during the Random Forest training component.")
        raise