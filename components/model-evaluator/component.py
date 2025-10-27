import argparse
import pandas as pd
import joblib
import os
import json
import traceback
from datetime import datetime
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
    config_bucket_name: str,
    config_blob: str,
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
    best_candidate_model_obj = None
    best_candidate_local_path = ""
    
    print("Evaluating candidate models using cross-validation on training data.")
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, path in models.items():
        model_file = os.path.join(path, "model.joblib")
        print(f"Loading model from {model_file}.")
        
        loaded_data = joblib.load(model_file)
        if isinstance(loaded_data, dict) and "model" in loaded_data:
            model = loaded_data["model"]
            if "label_encoder" in loaded_data:
                print(f"{name} - Quality classes: {loaded_data['label_encoder'].classes_}")
            else:
                print(f"{name} - No label encoder (uses original classes)")
        else:
            model = loaded_data
        
        scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring="accuracy")
        avg_score = scores.mean()
        print(f"{name} - Cross-Validation Accuracy: {avg_score:.4f}")
        if avg_score > best_candidate_cv_score:
            best_candidate_cv_score = avg_score
            best_candidate_name = name
            best_candidate_model_obj = model
            best_candidate_local_path = model_file
    
    print(f"Best candidate model: {best_candidate_name} with CV score {best_candidate_cv_score:.4f}")
    
    y_pred_candidate = best_candidate_model_obj.predict(X_test)
    candidate_accuracy = accuracy_score(y_test, y_pred_candidate)
    candidate_precision = precision_score(y_test, y_pred_candidate, average='weighted', zero_division=0)
    candidate_recall = recall_score(y_test, y_pred_candidate, average='weighted', zero_division=0)
    candidate_f1 = f1_score(y_test, y_pred_candidate, average='weighted', zero_division=0)
    
    print(f"Candidate Test Metrics - Accuracy: {candidate_accuracy:.4f}, Precision: {candidate_precision:.4f}, "
          f"Recall: {candidate_recall:.4f}, F1: {candidate_f1:.4f}")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(model_bucket_name)
    prod_blob = bucket.blob(prod_model_blob)
    
    decision = "keep_current"
    prod_metrics = None
    
    if prod_blob.exists():
        print("Production model exists. Downloading for comparison.")
        prod_local_path = "/tmp/prod_model.joblib"
        prod_blob.download_to_filename(prod_local_path)
        
        loaded_prod = joblib.load(prod_local_path)
        if isinstance(loaded_prod, dict) and "model" in loaded_prod:
            prod_model = loaded_prod["model"]
            if "label_encoder" in loaded_prod:
                print(f"Production model has label encoder with classes: {loaded_prod['label_encoder'].classes_}")
            else:
                print("Production model has no label encoder")
        else:
            prod_model = loaded_prod
        
        print("Production model loaded successfully.")
        
        y_pred_prod = prod_model.predict(X_test)
        prod_accuracy = accuracy_score(y_test, y_pred_prod)
        prod_precision = precision_score(y_test, y_pred_prod, average='weighted', zero_division=0)
        prod_recall = recall_score(y_test, y_pred_prod, average='weighted', zero_division=0)
        prod_f1 = f1_score(y_test, y_pred_prod, average='weighted', zero_division=0)
        
        prod_cv_scores = cross_val_score(prod_model, X_train, y_train, cv=cv_strategy, scoring="accuracy")
        prod_cv_score = prod_cv_scores.mean()
        
        prod_metrics = {
            "accuracy": float(prod_accuracy),
            "precision": float(prod_precision),
            "recall": float(prod_recall),
            "f1": float(prod_f1),
            "cv_score": float(prod_cv_score)
        }
        
        print(f"Production Test Metrics - Accuracy: {prod_accuracy:.4f}, Precision: {prod_precision:.4f}, "
              f"Recall: {prod_recall:.4f}, F1: {prod_f1:.4f}, CV: {prod_cv_score:.4f}")
        print(f"Candidate Test Metrics  - Accuracy: {candidate_accuracy:.4f}, Precision: {candidate_precision:.4f}, "
              f"Recall: {candidate_recall:.4f}, F1: {candidate_f1:.4f}, CV: {best_candidate_cv_score:.4f}")
        
        if candidate_accuracy > prod_accuracy:
            improvement = ((candidate_accuracy - prod_accuracy) / prod_accuracy) * 100
            print(f"Candidate model outperforms production by {improvement:.2f}%. Decision: DEPLOY NEW MODEL.")
            decision = "deploy_new"
        else:
            print("Production model is equal or better. Decision: KEEP CURRENT MODEL.")
            decision = "keep_current"
    else:
        print("No production model exists. Decision: DEPLOY NEW MODEL (first deployment).")
        decision = "deploy_new"
    
    if decision == "deploy_new":
        print("Uploading new model to production location.")
        prod_blob.upload_from_filename(best_candidate_local_path)
        new_model_uri = f"gs://{model_bucket_name}/{prod_model_blob}"
        print(f"New production model uploaded to: {new_model_uri}")
        
        print("Updating model configuration file.")
        config_bucket = storage_client.bucket(config_bucket_name)
        config_blob_obj = config_bucket.blob(config_blob)
        
        config_data = {
            "production_model_uri": new_model_uri,
            "production_model_updated_at": datetime.utcnow().isoformat() + "Z",
            "production_model_name": best_candidate_name,
            "production_model_accuracy": float(candidate_accuracy),
            "production_model_cv_score": float(best_candidate_cv_score)
        }
        
        config_blob_obj.upload_from_string(
            json.dumps(config_data, indent=2),
            content_type='application/json'
        )
        print(f"Configuration updated at gs://{config_bucket_name}/{config_blob}")
    else:
        print("Keeping current production model.")
        new_model_uri = f"gs://{model_bucket_name}/{prod_model_blob}"
    
    # Write outputs
    with open(decision_path, "w") as f:
        f.write(decision)
    with open(best_model_uri_path, "w") as f:
        f.write(new_model_uri)
    
    metrics_data = {
        "decision": decision,
        "candidate_model": {
            "name": best_candidate_name,
            "cv_score": float(best_candidate_cv_score),
            "test_accuracy": float(candidate_accuracy),
            "test_precision": float(candidate_precision),
            "test_recall": float(candidate_recall),
            "test_f1": float(candidate_f1)
        }
    }
    
    if prod_metrics:
        metrics_data["production_model"] = prod_metrics
        metrics_data["improvement"] = {
            "accuracy_delta": float(candidate_accuracy - prod_metrics["accuracy"]),
            "accuracy_improvement_percent": float(((candidate_accuracy - prod_metrics["accuracy"]) / prod_metrics["accuracy"]) * 100) if prod_metrics["accuracy"] > 0 else 0,
            "cv_score_delta": float(best_candidate_cv_score - prod_metrics["cv_score"])
        }
    else:
        metrics_data["production_model"] = None
        metrics_data["improvement"] = None
        metrics_data["note"] = "First model deployment - no production model to compare against"
    
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    
    print("Evaluation complete.")
    print(f"Final metrics summary:\n{json.dumps(metrics_data, indent=2)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data_path", type=str, required=True)
    parser.add_argument("--testing_data_path", type=str, required=True)
    parser.add_argument("--random_forest_model_path", type=str, required=True)
    parser.add_argument("--xgboost_model_path", type=str, required=True)
    parser.add_argument("--svm_model_path", type=str, required=True)
    parser.add_argument("--model_bucket_name", type=str, required=True)
    parser.add_argument("--prod_model_blob", type=str, required=True)
    parser.add_argument("--config_bucket_name", type=str, default="yannick-pipeline-root")
    parser.add_argument("--config_blob", type=str, default="config/model-config.json")
    parser.add_argument("--decision_path", type=str, required=True)
    parser.add_argument("--best_model_uri_path", type=str, required=True)
    parser.add_argument("--metrics_path", type=str, required=True)
    args = parser.parse_args()
    
    evaluate_and_decide(
        args.training_data_path,
        args.testing_data_path,
        args.random_forest_model_path,
        args.xgboost_model_path,
        args.svm_model_path,
        args.model_bucket_name,
        args.prod_model_blob,
        args.config_bucket_name,
        args.config_blob,
        args.decision_path,
        args.best_model_uri_path,
        args.metrics_path,
    )