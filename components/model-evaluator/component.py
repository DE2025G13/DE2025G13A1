import argparse
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score
from google.cloud import storage

def evaluate_and_decide(
    testing_data_path: str,
    decision_tree_model_path: str,
    linear_regression_model_path: str,
    logistic_regression_model_path: str,
    model_bucket_name: str,
    prod_model_blob: str,
    decision_path: str,
    best_model_uri_path: str
):
    """
    Evaluates candidate models, compares the best one to the production model,
    and outputs the decision ('deploy_new' or 'keep_old') and the best model's URI.
    """
    X_test = pd.read_csv(f"{testing_data_path}/x_test.csv")
    y_test = pd.read_csv(f"{testing_data_path}/y_test.csv").values.ravel()

    # Note: KFP passes the full GCS URI as the path for model artifacts
    models = {
        "decision_tree": decision_tree_model_path,
        "linear_regression": linear_regression_model_path,
        "logistic_regression": logistic_regression_model_path,
    }
    
    best_candidate_name = ""
    best_candidate_score = -1.0
    best_candidate_uri = ""

    # 1. Find the best model among the new candidates
    for name, uri in models.items():
        # The model path provided by KFP is a local file path inside the container
        model = joblib.load(uri)
        y_pred = model.predict(X_test)

        # For regression models, round predictions to match classification quality scores
        if name == "linear_regression":
             y_pred = [round(p) for p in y_pred]
        
        score = accuracy_score(y_test, y_pred)
        print(f"DEBUG: Candidate '{name}' accuracy: {score:.4f}")

        if score > best_candidate_score:
            best_candidate_score = score
            best_candidate_name = name
            best_candidate_uri = uri

    print(f"DEBUG: Best candidate is '{best_candidate_name}' with accuracy: {best_candidate_score:.4f}")

    # 2. Compare the best candidate against the current production model
    prod_score = -1.0
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(model_bucket_name)
        blob = bucket.blob(prod_model_blob)

        if blob.exists():
            local_prod_model = "/tmp/prod_model.joblib"
            blob.download_to_filename(local_prod_model)
            prod_model = joblib.load(local_prod_model)
            y_prod_pred = prod_model.predict(X_test)
            
            # Also round predictions for linear regression production models
            if isinstance(prod_model, (pd.DataFrame, pd.Series)) and not hasattr(prod_model, 'predict_proba'):
                 y_prod_pred = [round(p) for p in y_prod_pred]

            prod_score = accuracy_score(y_test, y_prod_pred)
            print(f"DEBUG: Production model accuracy: {prod_score:.4f}")
        else:
            print("DEBUG: No production model found. Any new model will be promoted.")
            prod_score = -0.1 # Ensure any model with non-negative score is better
    except Exception as e:
        print(f"DEBUG: Could not load or evaluate production model. Assuming no production model exists. Error: {e}")
        prod_score = -0.1

    # 3. Make the final decision and write outputs for the pipeline
    if best_candidate_score > prod_score:
        print(f"DECISION: New model '{best_candidate_name}' is better. Promoting.")
        decision = "deploy_new"
    else:
        print("DECISION: Production model is better or equal. Keeping old model.")
        decision = "keep_old"
        best_candidate_uri = "gs://none/none" # Provide a placeholder

    with open(decision_path, 'w') as f:
        f.write(decision)
    
    with open(best_model_uri_path, 'w') as f:
        f.write(best_candidate_uri)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testing_data_path', type=str, required=True)
    parser.add_argument('--decision_tree_model_path', type=str, required=True)
    parser.add_argument('--linear_regression_model_path', type=str, required=True)
    parser.add_argument('--logistic_regression_model_path', type=str, required=True)
    parser.add_argument('--model_bucket_name', type=str, required=True)
    parser.add_argument('--prod_model_blob', type=str, required=True)
    # These are special KFP output parameter paths
    parser.add_argument('--decision', type=str, required=True)
    parser.add_argument('--best_model_uri', type=str, required=True)
    args = parser.parse_args()

    evaluate_and_decide(
        args.testing_data_path,
        args.decision_tree_model_path,
        args.linear_regression_model_path,
        args.logistic_regression_model_path,
        args.model_bucket_name,
        args.prod_model_blob,
        args.decision,
        args.best_model_uri
    )