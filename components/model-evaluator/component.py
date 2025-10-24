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
    prod_model_blob: str
):
    X_test = pd.read_csv(f"{testing_data_path}/x_test.csv")
    y_test = pd.read_csv(f"{testing_data_path}/y_test.csv").values.ravel()

    models = {
        "decision_tree": decision_tree_model_path,
        "linear_regression": linear_regression_model_path,
        "logistic_regression": logistic_regression_model_path,
    }
    
    best_candidate_name = ""
    best_candidate_score = -1.0

    for name, path in models.items():
        model = joblib.load(path)
        y_pred = model.predict(X_test)

        if name == "linear_regression":
             y_pred = [round(p) for p in y_pred]
        
        score = accuracy_score(y_test, y_pred)
        print(f"DEBUG: Candidate '{name}' accuracy: {score}")
        if score > best_candidate_score:
            best_candidate_score = score
            best_candidate_name = name

    print(f"DEBUG: Best candidate is '{best_candidate_name}' with accuracy: {best_candidate_score}")

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
            
            if "predict_proba" not in dir(prod_model):
                 y_prod_pred = [round(p) for p in y_prod_pred]
            prod_score = accuracy_score(y_test, y_prod_pred)
            print(f"DEBUG: Production model accuracy: {prod_score}")
        else:
            print("DEBUG: No production model found.")
            prod_score = -0.1 
    except Exception as e:
        print(f"DEBUG: Could not load or evaluate production model. Error: {e}")

    if best_candidate_score > prod_score:
        print(best_candidate_name)
    else:
        print("keep_old")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testing_data_path', type=str, required=True)
    parser.add_argument('--decision_tree_model_path', type=str, required=True)
    parser.add_argument('--linear_regression_model_path', type=str, required=True)
    parser.add_argument('--logistic_regression_model_path', type=str, required=True)
    parser.add_argument('--model_bucket_name', type=str, required=True)
    parser.add_argument('--prod_model_blob', type=str, required=True)
    args = parser.parse_args()
    evaluate_and_decide(
        args.testing_data_path,
        args.decision_tree_model_path,
        args.linear_regression_model_path,
        args.logistic_regression_model_path,
        args.model_bucket_name,
        args.prod_model_blob
    )
