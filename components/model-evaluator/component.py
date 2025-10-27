import argparse
import pandas as pd
import joblib
import os
import json
import traceback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from google.cloud import storage
from sklearn.linear_model import LinearRegression
import xgboost as xgb

def evaluate_and_decide(
    training_data_path: str,
    testing_data_path: str,
    linear_regression_model_path: str,
    random_forest_model_path: str,
    xgboost_model_path: str,
    svm_model_path: str,
    model_bucket_name: str,
    prod_model_blob: str,
    decision_path: str,
    best_model_uri_path: str,
    metrics_path: str,
):
    """
    Evaluates models using K-Fold CV, logs comprehensive metrics for the best one, 
    compares it to production, and outputs the decision.
    """
    print("--- Starting Model Evaluation ---")
    
    # --- Load all necessary data ---
    print("Loading training and testing data...")
    X_train = pd.read_csv(os.path.join(training_data_path, "x_train.csv"))
    y_train = pd.read_csv(os.path.join(training_data_path, "y_train.csv")).values.ravel()
    X_test = pd.read_csv(os.path.join(testing_data_path, "x_test.csv"))
    y_test = pd.read_csv(os.path.join(testing_data_path, "y_test.csv")).values.ravel()
    print("Data loaded successfully.")

    models = {
        "linear_regression": linear_regression_model_path,
        "random_forest": random_forest_model_path,
        "xgboost": xgboost_model_path,
        "svm": svm_model_path,
    }
    
    best_candidate_name = ""
    best_candidate_cv_score = -1.0
    best_candidate_uri = ""
    metrics = {"scalar": []}

    # --- Stage 1: Select Best Candidate Model using K-Fold Cross-Validation ---
    print("\n--- Selecting Best Candidate via 5-Fold Cross-Validation on Training Data ---")
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_train_xgb_mapped = y_train - y_train.min() # Pre-map labels for XGBoost

    for name, path in models.items():
        print(f"Cross-validating candidate model '{name}'...")
        model = joblib.load(path)
        
        y_train_for_cv = y_train_xgb_mapped if name == "xgboost" else y_train

        scores = cross_val_score(model, X_train, y_train_for_cv, cv=cv_splitter, scoring='f1_weighted', n_jobs=-1)
        mean_score = scores.mean()
        
        print(f"-> Candidate '{name}' Mean CV F1-Score: {mean_score:.4f}")
        metrics["scalar"].append({"metric": f"{name}_mean_cv_f1", "value": mean_score})
        
        if mean_score > best_candidate_cv_score:
            best_candidate_cv_score = mean_score
            best_candidate_name = name
            best_candidate_uri = path 
    
    print(f"\nBest candidate from CV is '{best_candidate_name}' with Mean F1-Score: {best_candidate_cv_score:.4f}")
    
    # --- Stage 2: Evaluate the Best Candidate on the Hold-Out Test Set ---
    print(f"\n--- Evaluating '{best_candidate_name}' on Unseen Test Data ---")
    best_model = joblib.load(best_candidate_uri)
    
    if best_candidate_name == "xgboost":
        y_test_pred_raw = best_model.predict(X_test)
        y_test_pred = y_test_pred_raw + y_test.min()
    else:
        y_test_pred = best_model.predict(X_test)

    if best_candidate_name == "linear_regression":
        y_test_pred = [round(p) for p in y_test_pred]

    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

    print(f"-> Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    metrics["scalar"].extend([
        {"metric": "best_candidate_test_accuracy", "value": accuracy},
        {"metric": "best_candidate_test_precision", "value": precision},
        {"metric": "best_candidate_test_recall", "value": recall},
        {"metric": "best_candidate_test_f1", "value": f1}
    ])
    
    best_candidate_test_score = f1 # Use F1 for the final promotion decision

    # --- Stage 3: Evaluate Production Model and Make Final Decision ---
    print("\n--- Evaluating Production Model ---")
    prod_score = -1.0
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(model_bucket_name)
        blob = bucket.blob(prod_model_blob)
        if blob.exists():
            print(f"Production model found. Downloading...")
            local_prod_model_path = "/tmp/prod_model.joblib"
            blob.download_to_filename(local_prod_model_path)
            prod_model = joblib.load(local_prod_model_path)
            y_prod_pred = prod_model.predict(X_test)
            
            if isinstance(prod_model, LinearRegression): y_prod_pred = [round(p) for p in y_prod_pred]
            elif isinstance(prod_model, xgb.XGBClassifier): y_prod_pred = prod_model.predict(X_test) + y_test.min()

            prod_score = f1_score(y_test, y_prod_pred, average='weighted', zero_division=0)
            print(f"-> Production model F1-Score: {prod_score:.4f}")
            metrics["scalar"].append({"metric": "production_f1_score", "value": prod_score})
        else:
            print("No production model found. Promoting new model.")
    except Exception as e:
        print(f"Could not load or evaluate production model. Error: {e}")
        traceback.print_exc()

    print("\n--- Making Final Decision ---")
    decision = "keep_old"
    if best_candidate_test_score > prod_score:
        print(f"DECISION: New model '{best_candidate_name}' is better ({best_candidate_test_score:.4f} > {prod_score:.4f}). Promoting.")
        decision = "deploy_new"
    else:
        print(f"DECISION: Production model is better or equal. Keeping old model.")
        best_candidate_uri = "gs://none/none"

    print("\n--- Writing Output Artifacts ---")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f: json.dump(metrics, f)
    with open(decision_path, 'w') as f: f.write(decision)
    with open(best_model_uri_path, 'w') as f: f.write(best_candidate_uri)
    print("Output artifacts saved successfully.")
    print("\n--- Model Evaluation Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data_path', type=str, required=True)
    parser.add_argument('--testing_data_path', type=str, required=True)
    parser.add_argument('--linear_regression_model_path', type=str, required=True)
    parser.add_argument('--random_forest_model_path', type=str, required=True)
    parser.add_argument('--xgboost_model_path', type=str, required=True)
    parser.add_argument('--svm_model_path', type=str, required=True)
    parser.add_argument('--model_bucket_name', type=str, required=True)
    parser.add_argument('--prod_model_blob', type=str, required=True)
    parser.add_argument('--decision', type=str, required=True)
    parser.add_argument('--best_model_uri', type=str, required=True)
    parser.add_argument('--metrics', type=str, required=True)
    args = parser.parse_args()
    evaluate_and_decide(**vars(args))