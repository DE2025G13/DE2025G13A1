import argparse
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os

def train_model(training_data_path: str, model_artifact_path: str):
    print("Starting SVM training process.")
    X_train = pd.read_csv(os.path.join(training_data_path, "x_train.csv"))
    y_train = pd.read_csv(os.path.join(training_data_path, "y_train.csv"))
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, probability=True, random_state=42))
    ])
    print("Fitting the SVM pipeline to the training data.")
    pipeline.fit(X_train, y_train.values.ravel())
    print("SVM pipeline fitting has completed.")
    
    os.makedirs(model_artifact_path, exist_ok=True)
    model_file_path = os.path.join(model_artifact_path, "model.joblib")
    joblib.dump(pipeline, model_file_path)
    print(f"SVM model pipeline has been saved to {model_file_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data_path", type=str, required=True)
    parser.add_argument("--model_artifact_path", type=str, required=True)
    args = parser.parse_args()
    try:
        train_model(args.training_data_path, args.model_artifact_path)
    except Exception as e:
        print("An error occurred during the SVM training component.")
        raise