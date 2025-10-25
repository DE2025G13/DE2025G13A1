import argparse
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

def train_model(training_data_path: str, model_artifact_path: str):
    X_train = pd.read_csv(os.path.join(training_data_path, "x_train.csv"))
    y_train = pd.read_csv(os.path.join(training_data_path, "y_train.csv"))
    
    model = DecisionTreeClassifier(max_depth=10, random_state=42)
    model.fit(X_train, y_train.values.ravel())
    
    os.makedirs(os.path.dirname(model_artifact_path), exist_ok=True)
    
    # KFP will treat any file saved in model_artifact_path as the model artifact
    joblib.dump(model, model_artifact_path)
    print(f"Decision Tree model saved to {model_artifact_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data_path', type=str, required=True)
    parser.add_argument('--model_artifact_path', type=str, required=True)
    args = parser.parse_args()
    train_model(args.training_data_path, args.model_artifact_path)