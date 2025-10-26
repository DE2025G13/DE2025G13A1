import argparse
import pandas as pd
import joblib
import os
import json
import traceback
from sklearn.metrics import accuracy_score
from google.cloud import storage
from sklearn.linear_model import LinearRegression

def evaluate_and_decide(
    testing_data_path: str,
    decision_tree_model_path: str,
    linear_regression_model_path: str,
    logistic_regression_model_path: str,
    model_bucket_name: str,
    prod_model_blob: str,
    decision_path: str,
    best_model_uri_path: str,
    metrics_path: str,
):
    """
    Evaluates models, logs metrics, compares to production, and outputs the decision.
    """
    # ... (The first part of the function remains the same)
    print("--- Starting Model Evaluation ---")
    
    print(f"Loading testing data from: {testing_data_path}")
    X_test = pd.read_csv(os.path.join(testing_data_path, "x_test.csv"))
    y_test = pd.read_csv(os.path.join(testing_data_path, "y_test.csv")).values.ravel()
    print("Testing data loaded successfully.")

    models = {
        "decision_tree": decision_tree_model_path,
        "linear_regression": linear_regression_model_path,
        "logistic_regression": logistic_regression_model_path,
    }
    
    best_candidate_name = ""
    best_candidate_score = -1.0
    best_candidate_uri = ""
    metrics = {"scalar": []}

    print("\n--- Evaluating Candidate Models ---")
    for name, path in models.items():
        print(f"Loading candidate model '{name}' from path: {path}")
        model = joblib.load(path)
        y_pred = model.predict(X_test)
        if name == "linear_regression":
             y_pred = [round(p) for p in y_pred]
        score = accuracy_score(y_test, y_pred)
        print(f"-> Candidate '{name}' accuracy: {score:.4f}")
        metrics["scalar"].append({"metric": f"{name}_accuracy", "value": score})
        if score > best_candidate_score:
            best_candidate_score = score
            best_candidate_name = name
            best_candidate_uri = path 
    
    print(f"\nBest candidate is '{best_candidate_name}' with accuracy: {best_candidate_score:.4f}")
    metrics["scalar"].append({"metric": "best_candidate_accuracy", "value": best_candidate_score})

    print("\n--- Evaluating Production Model ---")
    prod_score = -1.0
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(model_bucket_name)
        blob = bucket.blob(prod_model_blob)
        if blob.exists():
            print(f"Production model found at gs://{model_bucket_name}/{prod_model_blob}. Downloading...")
            local_prod_model_path = "/tmp/prod_model.joblib"
            blob.download_to_filename(local_prod_model_path)
            prod_model = joblib.load(local_prod_model_path)
            y_prod_pred = prod_model.predict(X_test)
            if isinstance(prod_model, LinearRegression):
                 y_prod_pred = [round(p) for p in y_prod_pred]
            prod_score = accuracy_score(y_test, y_prod_pred)
            print(f"-> Production model accuracy: {prod_score:.4f}")
            metrics["scalar"].append({"metric": "production_accuracy", "value": prod_score})
        else:
            print("No production model found. Any new model will be promoted.")
    except Exception as e:
        print(f"Could not load or evaluate production model. Assuming it doesn't exist. Error: {e}")

    print("\n--- Making Final Decision ---")
    decision = "keep_old"
    if best_candidate_score > prod_score:
        print(f"DECISION: New model '{best_candidate_name}' is better. Promoting.")
        decision = "deploy_new"
    else:
        print(f"DECISION: Production model is better or equal. Keeping old model.")
        best_candidate_uri = "gs://none/none"

    # --- THIS IS THE FIX ---
    # Ensure the directories for ALL output files exist before writing to them.
    print("\n--- Writing Output Artifacts ---")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    os.makedirs(os.path.dirname(decision_path), exist_ok=True)
    os.makedirs(os.path.dirname(best_model_uri_path), exist_ok=True)
    # -----------------------

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    print(f"Metrics saved to {metrics_path}")

    with open(decision_path, 'w') as f:
        f.write(decision)
    print(f"Decision saved to {decision_path}")
    
    with open(best_model_uri_path, 'w') as f:
        f.write(best_candidate_uri)
    print(f"Best model URI saved to {best_model_uri_path}")
    
    print("\n--- Model Evaluation Finished Successfully ---")


if __name__ == '__main__':
    # ... (The argparse section remains exactly the same)
    parser = argparse.ArgumentParser()
    parser.add_argument('--testing_data_path', type=str, required=True)
    parser.add_argument('--decision_tree_model_path', type=str, required=True)
    parser.add_argument('--linear_regression_model_path', type=str, required=True)
    parser.add_argument('--logistic_regression_model_path', type=str, required=True)
    parser.add_argument('--model_bucket_name', type=str, required=True)
    parser.add_argument('--prod_model_blob', type=str, required=True)
    parser.add_argument('--decision', type=str, required=True)
    parser.add_argument('--best_model_uri', type=str, required=True)
    parser.add_argument('--metrics', type=str, required=True)
    args = parser.parse_args()

    # The robust try/except block remains the same
    try:
        evaluate_and_decide(
            args.testing_data_path,
            args.decision_tree_model_path,
            args.linear_regression_model_path,
            args.logistic_regression_model_path,
            args.model_bucket_name,
            args.prod_model_blob,
            args.decision,
            args.best_model_uri,
            args.metrics,
        )
    except Exception:
        print("--- ERROR in Model Evaluator ---")
        error_message = traceback.format_exc()
        print(error_message)
        print("---------------------------------")
        raise