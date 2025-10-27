import argparse
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

def train_model(training_data_path: str, model_artifact_path: str):
    """Trains a Support Vector Classifier model with scaling."""
    print("--- Starting SVM Training ---")
    X_train = pd.read_csv(os.path.join(training_data_path, "x_train.csv"))
    y_train = pd.read_csv(os.path.join(training_data_path, "y_train.csv"))
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train.values.ravel())
    
    os.makedirs(os.path.dirname(model_artifact_path), exist_ok=True)
    
    joblib.dump(pipeline, model_artifact_path)
    print(f"SVM model pipeline saved to {model_artifact_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data_path', type=str, required=True)
    parser.add_argument('--model_artifact_path', type=str, required=True)
    args = parser.parse_args()
    train_model(args.training_data_path, args.model_artifact_path)