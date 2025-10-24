import argparse
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from google.cloud import storage
import os

def evaluate_and_decide(
    processed_data_path: str,
    decision_tree_model_path: str,
    linear_regression_model_path: str,
    logistic_regression_model_path: str,
    model_bucket_name: str,
    prod_model_blob: str,
    output_path: str  # <-- Add this new argument
):
    """
    Evaluates models and WRITES the GCS URI of the best model to a file
    if it's better than production, otherwise WRITES 'keep_old'.
    """
    X_test = pd.read_csv(f"{processed_data_path}/x_test.csv")
    y_test = pd.read_csv(f"{processed_data_path}/y_test.csv").values.ravel()

    models = {
        "decision_tree": decision_tree_model_path,
        "linear_regression": linear_regression_model_path,
        "logistic_regression": logistic_regression_model_path,
    }
    
    best_model_name = ""
    best_model_score = -1.0

    for name, path in models.items():
        model = joblib.load(path)
        y_pred = model.predict(X_test)

        if name == "linear_regression":
             y_pred = [round(p) for p in y_pred]
        
        score = accuracy_score(y_test, y_pred)
        print(f"DEBUG: Model '{name}' accuracy: {score}")

        if score > best_model_score:
            best_model_score = score
            best_model_name = name

    print(f"DEBUG: Best candidate is '{best_model_name}' with accuracy: {best_model_score}")

    storage_client = storage.Client()
    prod_score = -1.0
    try:
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
    except Exception as e:
        print(f"DEBUG: Could not load or evaluate production model. Error: {e}")

    result_to_write = "keep_old"
    if best_model_score > prod_score:
        result_to_write = models[best_model_name].replace("/gcs/", "gs://")
    
    # Instead of printing, write the result to the specified output file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(result_to_write)
    print(f"Wrote '{result_to_write}' to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-data-path', type=str, required=True)
    parser.add_argument('--decision-tree-model-path', type=str, required=True)
    parser.add_argument('--linear-regression-model-path', type=str, required=True)
    parser.add_argument('--logistic-regression-model-path', type=str, required=True)
    parser.add_argument('--model-bucket-name', type=str, required=True)
    parser.add_argument('--prod-model-blob', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True) # <-- Add argument parsing
    
    args = parser.parse_args()
    evaluate_and_decide(
        args.processed_data_path,
        args.decision_tree_model_path,
        args.linear_regression_model_path,
        args.logistic_regression_model_path,
        args.model_bucket_name,
        args.prod_model_blob,
        args.output_path # <-- Pass the new argument
    )