import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import traceback # Import the traceback module for detailed error logging

def train_model(training_data_path: str, model_artifact_path: str):
    """
    Trains a RandomForestClassifier model in a robust, memory-safe manner.
    """
    print("--- Starting Random Forest Training ---")
    
    # Load training data
    X_train = pd.read_csv(os.path.join(training_data_path, "x_train.csv"))
    y_train = pd.read_csv(os.path.join(training_data_path, "y_train.csv"))
    
    # Initialize the model. We are not setting n_jobs, so it defaults to 1 (single-core).
    model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=10)
    
    # Fit the model
    print("Fitting the Random Forest model...")
    model.fit(X_train, y_train.values.ravel())
    print("Model fitting complete.")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(model_artifact_path), exist_ok=True)
    
    # Save the trained model artifact
    joblib.dump(model, model_artifact_path)
    print(f"Random Forest model successfully saved to {model_artifact_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data_path', type=str, required=True, help='Path to the folder containing training data.')
    parser.add_argument('--model_artifact_path', type=str, required=True, help='Path to save the output model artifact.')
    args = parser.parse_args()
    
    try:
        train_model(args.training_data_path, args.model_artifact_path)
    except Exception as e:
        print("--- ERROR IN RANDOM FOREST TRAINER ---")
        traceback.print_exc()
        print("--------------------------------------")
        raise