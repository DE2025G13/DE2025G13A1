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
    random_forest_model_path: str,
    xgboost_model_path: str,
    svm_model_path: str,
    model_bucket_name: str,
    prod_model_blob: str,
    decision_path: str,
    best_model_uri_path: str,
    metrics_path: str,
):
    print("Starting the model evaluation process.")
    print("Loading training and testing datasets.")
    X_train = pd.read_csv(os.path.join(training_data_path, "x_train.csv"))
    y_train = pd.read_csv(os.path.join(training_data_path, "y_train.csv")).values.ravel()
    X_test = pd.read_csv(os.path.join(testing_data_path, "x_test.csv"))
    y_test = pd.read_csv(os.path.join(testing_data_path, "y_test.csv")).values.ravel()
    print("Data loading completed.")
    models = {
        "random_forest": random_forest_model_path,
        "xgboost": xgboost_model_path,
        "svm": svm_model_path,
    }
    best_candidate_name = ""
    best_candidate_cv_score = -1.0
    best_candidate_uri = ""
    metrics = {"scalar": []}
    print("Selecting the best candidate model using 5-fold cross-validation.")
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_train_xgb_mapped = y_train - y_train.min()
    for name, path in models.items():
        print(f"Cross-validating candidate model: {name}.")
        model = joblib.load(path)
        y_train_for_cv = y_train_xgb_mapped if name == "xgboost" else y_train
        scores = cross_val_score(model, X_train, y_train_for_cv, cv=cv_splitter, scoring="f1_weighted", n_jobs=-1)
        mean_score = scores.mean()
        print(f"-> Candidate '{name}' had a Mean CV F1-Score of: {mean_score:.4f}")
        metrics["scalar"].append({"metric": f"{name}_mean_cv_f1", "value": mean_score})
        if mean_score > best_candidate_cv_score:
            best_candidate_cv_score = mean_score
            best_candidate_name = name
            best_candidate_uri = path
    print(f"The best candidate from cross-validation is '{best_candidate_name}' with a score of {best_candidate_cv_score:.4f}.")
    print(f"Evaluating the best candidate, '{best_candidate_name}', on the hold-out test set.")
    best_model = joblib.load(best_candidate_uri)
    if best_candidate_name == "xgboost":
        y_test_pred_raw = best_model.predict(X_test)
        y_test_pred = y_test_pred_raw + y_test.min()
    else:
        y_test_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_test_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)
    print(f"-> Test Metrics: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1:.4f}")
    metrics["scalar"].extend([
        {"metric": "best_candidate_test_accuracy", "value": accuracy},
        {"metric": "best_candidate_test_precision", "value": precision},
        {"metric": "best_candidate_test_recall", "value": recall},
        {"metric": "best_candidate_test_f1", "value": f1}
    ])
    best_candidate_test_score = f1
    print("Evaluating the current production model for comparison.")
    prod_score = -1.0
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(model_bucket_name)
        blob = bucket.blob(prod_model_blob)
        if blob.exists():
            print("Production model found, downloading for evaluation.")
            local_prod_model_path = "/tmp/prod_model.joblib"
            blob.download_to_filename(local_prod_model_path)
            prod_model = joblib.load(local_prod_model_path)
            y_prod_pred = prod_model.predict(X_test)
            if isinstance(prod_model, xgb.XGBClassifier): y_prod_pred = prod_model.predict(X_test) + y_test.min()
            prod_score = f1_score(y_test, y_prod_pred, average="weighted", zero_division=0)
            print(f"-> Production model F1-Score on test set: {prod_score:.4f}")
            metrics["scalar"].append({"metric": "production_f1_score", "value": prod_score})
        else:
            print("No production model found. The new model will be promoted by default.")
    except Exception as e:
        print(f"Could not load or evaluate production model, assuming it does not exist. Error: {e}")
        traceback.print_exc()
    print("Making the final promotion decision.")
    decision = "keep_old"
    if best_candidate_test_score > prod_score:
        print(f"Decision: New model '{best_candidate_name}' is better and will be promoted.")
        decision = "deploy_new"
    else:
        print("Decision: The current production model is better or equal, so it will be kept.")
        best_candidate_uri = "gs://none/none"
    print("Writing decision and metric artifacts.")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f: json.dump(metrics, f)
    with open(decision_path, "w") as f: f.write(decision)
    with open(best_model_uri_path, "w") as f: f.write(best_candidate_uri)
    print("Output artifacts have been saved successfully.")
    print("Model evaluation process finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data_path", type=str, required=True)
    parser.add_argument("--testing_data_path", type=str, required=True)
    parser.add_argument("--random_forest_model_path", type=str, required=True)
    parser.add_argument("--xgboost_model_path", type=str, required=True)
    parser.add_argument("--svm_model_path", type=str, required=True)
    parser.add_argument("--model_bucket_name", type=str, required=True)
    parser.add_argument("--prod_model_blob", type=str, required=True)
    parser.add_argument("--decision_path", type=str, required=True)
    parser.add_argument("--best_model_uri_path", type=str, required=True)
    parser.add_argument("--metrics_path", type=str, required=True)
    args = parser.parse_args()
    evaluate_and_decide(**vars(args))