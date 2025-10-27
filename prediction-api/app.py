from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import json
from google.cloud import storage
import traceback

app = Flask(__name__)
CONFIG_BUCKET = os.environ.get("CONFIG_BUCKET", "yannick-pipeline-root")
CONFIG_BLOB = os.environ.get("CONFIG_BLOB", "config/model-config.json")
LOCAL_MODEL_PATH = "/tmp/model.joblib"
model = None

def get_production_model_uri():
    """Fetch the production model URI from the config file in GCS"""
    try:
        print(f"Fetching model config from gs://{CONFIG_BUCKET}/{CONFIG_BLOB}")
        storage_client = storage.Client()
        bucket = storage_client.bucket(CONFIG_BUCKET)
        blob = bucket.blob(CONFIG_BLOB)
        config_str = blob.download_as_text()
        config = json.loads(config_str)
        model_uri = config.get("production_model_uri")
        print(f"Production model URI: {model_uri}")
        return model_uri
    except Exception as e:
        print(f"Error fetching model config: {e}")
        traceback.print_exc()
        return None

def download_model():
    print("Attempting to download the production model.")
    model_uri = get_production_model_uri()
    if not model_uri:
        print("Fatal Error: Could not determine production model URI.")
        return False
    
    # Parse gs://bucket/path/to/model.joblib
    if not model_uri.startswith("gs://"):
        print(f"Invalid model URI: {model_uri}")
        return False
    
    path_parts = model_uri.replace("gs://", "").split("/", 1)
    bucket_name = path_parts[0]
    blob_path = path_parts[1]
    
    print(f"Downloading from gs://{bucket_name}/{blob_path}")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.download_to_filename(LOCAL_MODEL_PATH)
        print("Model download was successful.")
        return True
    except Exception as e:
        print("A critical error occurred during model download.")
        traceback.print_exc()
        return False

def load_model():
    global model
    if os.path.exists(LOCAL_MODEL_PATH):
        try:
            print("Loading model from local file into memory.")
            model = joblib.load(LOCAL_MODEL_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"An error occurred while loading the model file: {e}")
            model = None
    else:
        print("Model file could not be found locally.")
        model = None

if download_model():
    load_model()

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model is not available or failed to load. Please check server logs."}), 503
    try:
        data = request.get_json()
        print(f"Received prediction request with data: {data}")
        features = [
            "type",
            "fixed_acidity",
            "volatile_acidity",
            "citric_acid",
            "residual_sugar",
            "chlorides",
            "free_sulfur_dioxide",
            "total_sulfur_dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol"
        ]
        input_df = pd.DataFrame([data], columns=features)
        input_df["type"] = input_df["type"].apply(lambda x: 1 if x == "red" else 0)
        print(f"Encoded input for model: {input_df.to_dict('records')}")
        prediction_result = model.predict(input_df)
        final_prediction = int(round(prediction_result[0]))
        print(f"Model prediction result: {final_prediction}")
        return jsonify({"prediction": final_prediction})
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred during the prediction process."}), 400

@app.route("/health", methods=["GET"])
def health_check():
    if model:
        return jsonify({"status": "ok", "model": "loaded"}), 200
    else:
        print("Health check failed: model not loaded. Attempting to reload.")
        if download_model():
            load_model()
            if model:
                return jsonify({"status": "ok", "model": "reloaded"}), 200
        return jsonify({"status": "error", "model": "not_loaded"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))