import argparse
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score
from google.cloud import storage

def download_gcs_file(uri, local_path):
    """Downloads a single file from GCS."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    storage_client = storage.Client()
    bucket_name, blob_name = uri.replace("gs://", "").split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    bucket.blob(blob_name).download_to_filename(local_path)
    print(f"Downloaded {uri} to {local_path}")

def download_gcs_dir(uri, local_dir):
    """Downloads all files from a GCS prefix."""
    os.makedirs(local_dir, exist_ok=True)
    storage_client = storage.Client()
    bucket_name, prefix = uri.replace("gs://", "").split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
    for blob in blobs:
        if not blob.name.endswith('/'):
            blob.download_to_filename(os.path.join(local_dir, os.path.basename(blob.name)))

def evaluate_and_decide(
    processed_data_uri: str,
    decision_tree_model_uri: str,
    linear_regression_model_uri: str,
    logistic_regression_model_uri: str,
    model_bucket_name: str,
    prod_model_blob: str
):
    """
    Evaluates models and PRINTS the GCS URI of the best model.
    """
    local_data_path = "/tmp/data"
    download_gcs_dir(processed_data_uri, local_data_path)
    X_test = pd.read_csv(f"{local_data_path}/x_test.csv")
    y_test = pd.read_csv(f"{local_data_path}/y_test.csv").values.ravel()

    models = {
        decision_tree_model_uri: "/tmp/dt_model.joblib",
        linear_regression_model_uri: "/tmp/lr_model.joblib",
        logistic_regression_model_uri: "/tmp/logr_model.joblib",
    }
    
    for uri, path in models.items():
        download_gcs_file(uri, path)

    best_model_uri = ""
    best_model_score = -1.0

    for uri, path in models.items():
        model = joblib.load(path)
        y_pred = model.predict(X_test)

        if "linear-regression" in uri:
             y_pred = [round(p) for p in y_pred]
        
        score = accuracy_score(y_test, y_pred)
        print(f"DEBUG: Candidate '{uri}' accuracy: {score}")

        if score > best_model_score:
            best_model_score = score
            best_model_uri = uri

    print(f"DEBUG: Best candidate is '{best_model_uri}' with accuracy: {best_model_score}")

    prod_score = -1.0
    try:
        prod_model_uri = f"gs://{model_bucket_name}/{prod_model_blob}"
        local_prod_model = "/tmp/prod_model.joblib"
        download_gcs_file(prod_model_uri, local_prod_model)
        prod_model = joblib.load(local_prod_model)
        y_prod_pred = prod_model.predict(X_test)
        
        if "predict_proba" not in dir(prod_model):
             y_prod_pred = [round(p) for p in y_prod_pred]
        prod_score = accuracy_score(y_test, y_prod_pred)
        print(f"DEBUG: Production model accuracy: {prod_score}")
    except Exception as e:
        print(f"DEBUG: No production model found or could not evaluate. Error: {e}")

    if best_model_score > prod_score:
        print(best_model_uri)
    else:
        print("keep_old")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-data-uri', type=str, required=True)
    parser.add_argument('--decision-tree-model-uri', type=str, required=True)
    parser.add_argument('--linear-regression-model-uri', type=str, required=True)
    parser.add_argument('--logistic-regression-model-uri', type=str, required=True)
    parser.add_argument('--model-bucket-name', type=str, required=True)
    parser.add_argument('--prod-model-blob', type=str, required=True)
    args = parser.parse_args()
    evaluate_and_decide(
        args.processed_data_uri,
        args.decision_tree_model_uri,
        args.linear_regression_model_uri,
        args.logistic_regression_model_uri,
        args.model_bucket_name,
        args.prod_model_blob
    )