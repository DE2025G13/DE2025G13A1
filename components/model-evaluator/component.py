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
    
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_candidates_metrics = {}
    best_candidate_name = ""
    best_candidate_cv_score = -1.0
    best_candidate_model_obj = None
    best_candidate_local_path = ""
    
    print("Evaluating candidate models using cross-validation on training data.")
    
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
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring="accuracy")
        avg_cv_score = cv_scores.mean()
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        test_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        test_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        all_candidates_metrics[name] = {
            "cv_score_mean": float(avg_cv_score),
            "cv_score_std": float(cv_scores.std()),
            "cv_scores_per_fold": [float(s) for s in cv_scores],
            "test_accuracy": float(test_accuracy),
            "test_precision": float(test_precision),
            "test_recall": float(test_recall),
            "test_f1": float(test_f1),
            "model_file": model_file
        }
        print(f"{name} - CV Accuracy: {avg_cv_score:.4f} (±{cv_scores.std():.4f}), Test Accuracy: {test_accuracy:.4f}")
        if avg_cv_score > best_candidate_cv_score:
            best_candidate_cv_score = avg_cv_score
            best_candidate_name = name
            best_candidate_model_obj = model
            best_candidate_local_path = model_file
    
    print(f"Best candidate model: {best_candidate_name} with CV score {best_candidate_cv_score:.4f}")
    best_metrics = all_candidates_metrics[best_candidate_name]
    
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
            "cv_score_mean": float(prod_cv_score),
            "cv_score_std": float(prod_cv_scores.std()),
            "test_accuracy": float(prod_accuracy),
            "test_precision": float(prod_precision),
            "test_recall": float(prod_recall),
            "test_f1": float(prod_f1)
        }
        print(f"Production Metrics - CV: {prod_cv_score:.4f} (±{prod_cv_scores.std():.4f}), Test Accuracy: {prod_accuracy:.4f}")
        print(f"Candidate Metrics  - CV: {best_candidate_cv_score:.4f} (±{best_metrics['cv_score_std']:.4f}), Test Accuracy: {best_metrics['test_accuracy']:.4f}")
        if best_metrics['test_accuracy'] > prod_accuracy:
            improvement = ((best_metrics['test_accuracy'] - prod_accuracy) / prod_accuracy) * 100
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
            "production_model_accuracy": float(best_metrics['test_accuracy']),
            "production_model_cv_score": float(best_candidate_cv_score)
        }
        config_blob_obj.upload_from_string(json.dumps(config_data, indent=2), content_type='application/json')
        print(f"Configuration updated at gs://{config_bucket_name}/{config_blob}")
    else:
        print("Keeping current production model.")
        new_model_uri = f"gs://{model_bucket_name}/{prod_model_blob}"
    
    os.makedirs(os.path.dirname(decision_path), exist_ok=True)
    os.makedirs(os.path.dirname(best_model_uri_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(decision_path, "w") as f:
        f.write(decision)
    with open(best_model_uri_path, "w") as f:
        f.write(new_model_uri)
    
    metrics_data = {
        "decision": decision,
        "evaluation_timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset_info": {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "n_features": X_train.shape[1],
            "quality_classes": sorted([int(q) for q in set(y_train)])
        },
        "selected_model": {
            "name": best_candidate_name,
            "cv_score_mean": float(best_candidate_cv_score),
            "cv_score_std": float(best_metrics['cv_score_std']),
            "cv_scores_per_fold": best_metrics['cv_scores_per_fold'],
            "test_accuracy": float(best_metrics['test_accuracy']),
            "test_precision": float(best_metrics['test_precision']),
            "test_recall": float(best_metrics['test_recall']),
            "test_f1": float(best_metrics['test_f1'])
        },
        "all_candidates": {
            name: {
                "cv_score_mean": metrics["cv_score_mean"],
                "cv_score_std": metrics["cv_score_std"],
                "cv_scores_per_fold": metrics["cv_scores_per_fold"],
                "test_accuracy": metrics["test_accuracy"],
                "test_precision": metrics["test_precision"],
                "test_recall": metrics["test_recall"],
                "test_f1": metrics["test_f1"],
                "selected": (name == best_candidate_name)
            }
            for name, metrics in all_candidates_metrics.items()
        },
        "model_ranking": sorted(
            [{"model": name, "cv_score": metrics["cv_score_mean"], "test_accuracy": metrics["test_accuracy"]} 
             for name, metrics in all_candidates_metrics.items()],
            key=lambda x: x["cv_score"],
            reverse=True
        )
    }
    
    if prod_metrics:
        metrics_data["production_model"] = prod_metrics
        metrics_data["improvement"] = {
            "cv_score_delta": float(best_candidate_cv_score - prod_metrics["cv_score_mean"]),
            "cv_score_improvement_percent": float(((best_candidate_cv_score - prod_metrics["cv_score_mean"]) / prod_metrics["cv_score_mean"]) * 100) if prod_metrics["cv_score_mean"] > 0 else 0,
            "test_accuracy_delta": float(best_metrics['test_accuracy'] - prod_metrics["test_accuracy"]),
            "test_accuracy_improvement_percent": float(((best_metrics['test_accuracy'] - prod_metrics["test_accuracy"]) / prod_metrics["test_accuracy"]) * 100) if prod_metrics["test_accuracy"] > 0 else 0,
            "test_f1_delta": float(best_metrics['test_f1'] - prod_metrics["test_f1"])
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