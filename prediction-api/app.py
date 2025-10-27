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
FALLBACK_MODEL_URI = "gs://yannick-wine-models/production_model/model.joblib"
LOCAL_MODEL_PATH = "/tmp/model.joblib"
model = None
model_metadata = {}

def get_production_model_uri():
    """Fetch the production model URI from the config file in GCS"""
    try:
        print(f"Fetching model config from gs://{CONFIG_BUCKET}/{CONFIG_BLOB}")
        storage_client = storage.Client()
        bucket = storage_client.bucket(CONFIG_BUCKET)
        blob = bucket.blob(CONFIG_BLOB)
        
        if not blob.exists():
            print(f"Config blob does not exist! Using fallback: {FALLBACK_MODEL_URI}")
            return FALLBACK_MODEL_URI
        
        config_str = blob.download_as_text()
        print(f"Downloaded config (length={len(config_str)})")
        
        if not config_str or config_str.strip() == "":
            print(f"Config is empty! Using fallback: {FALLBACK_MODEL_URI}")
            return FALLBACK_MODEL_URI
        
        config = json.loads(config_str)
        model_uri = config.get("production_model_uri")
        print(f"Production model URI from config: {model_uri}")
        return model_uri
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Using fallback: {FALLBACK_MODEL_URI}")
        return FALLBACK_MODEL_URI
    except Exception as e:
        print(f"Error fetching model config: {e}")
        traceback.print_exc()
        print(f"Using fallback: {FALLBACK_MODEL_URI}")
        return FALLBACK_MODEL_URI

def download_model():
    print("Attempting to download the production model.")
    model_uri = get_production_model_uri()
    if not model_uri:
        print("Fatal Error: Could not determine production model URI.")
        return False
    
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
        
        if not blob.exists():
            print(f"ERROR: Model blob does not exist at {model_uri}")
            return False
        
        blob.download_to_filename(LOCAL_MODEL_PATH)
        print("Model download was successful.")
        return True
    except Exception as e:
        print("A critical error occurred during model download.")
        traceback.print_exc()
        return False

def load_model():
    global model, model_metadata
    if os.path.exists(LOCAL_MODEL_PATH):
        try:
            print("Loading model from local file into memory.")
            loaded_data = joblib.load(LOCAL_MODEL_PATH)
            
            if isinstance(loaded_data, dict) and "model" in loaded_data:
                model = loaded_data["model"]
                model_metadata = {
                    "label_encoder": loaded_data.get("label_encoder", None),
                    "model_type": loaded_data.get("model_type", "unknown")
                }
                if model_metadata["label_encoder"] is not None:
                    print(f"Model loaded successfully. Type: {model_metadata['model_type']}, Classes: {model_metadata['label_encoder'].classes_}")
                else:
                    print(f"Model loaded successfully. Type: {model_metadata['model_type']}")
            else:
                model = loaded_data
                model_metadata = {
                    "label_encoder": None,
                    "model_type": "legacy"
                }
                print("Model loaded (legacy format without metadata).")
        except Exception as e:
            print(f"An error occurred while loading the model file: {e}")
            traceback.print_exc()
            model = None
            model_metadata = {}
    else:
        print("Model file could not be found locally.")
        model = None
        model_metadata = {}

if download_model():
    load_model()

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model is not available. Please run the training pipeline first."}), 503
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
        encoded_prediction = int(prediction_result[0])
        label_encoder = model_metadata.get("label_encoder")
        if label_encoder is not None:
            final_prediction = int(label_encoder.inverse_transform([encoded_prediction])[0])
            print(f"Encoded prediction: {encoded_prediction}, Decoded: {final_prediction}")
        else:
            final_prediction = encoded_prediction
            print(f"Direct prediction: {final_prediction}")
        
        final_prediction = max(0, min(10, final_prediction))
        
        return jsonify({"prediction": final_prediction})
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred during the prediction process."}), 400

@app.route("/health", methods=["GET"])
def health_check():
    if model:
        label_encoder = model_metadata.get("label_encoder")
        health_info = {
            "status": "ok", 
            "model": "loaded",
            "model_type": model_metadata.get("model_type", "unknown")
        }
        if label_encoder is not None:
            health_info["quality_classes"] = label_encoder.classes_.tolist()
        return jsonify(health_info), 200
    else:
        print("Health check failed: model not loaded. Attempting to reload.")
        if download_model():
            load_model()
            if model:
                return jsonify({"status": "ok", "model": "reloaded"}), 200
        return jsonify({"status": "error", "model": "not_loaded"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))