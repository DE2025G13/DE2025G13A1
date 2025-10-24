import argparse
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

def train_model(training_data_path: str, model_artifact_path: str):
    X_train = pd.read_csv(f"{training_data_path}/x_train.csv")
    y_train = pd.read_csv(f"{training_data_path}/y_train.csv")
    model = LinearRegression()
    model.fit(X_train, y_train.values.ravel())
    os.makedirs(os.path.dirname(model_artifact_path), exist_ok=True)
    joblib.dump(model, model_artifact_path)
    print(f"Linear Regression model saved to {model_artifact_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data_path', type=str, required=True)
    parser.add_argument('--model_artifact_path', type=str, required=True)
    args = parser.parse_args()
    train_model(args.training_data_path, args.model_artifact_path)
