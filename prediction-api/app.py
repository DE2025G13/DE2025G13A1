from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
from google.cloud import storage
import traceback

app = Flask(__name__)
MODEL_BUCKET_NAME = os.environ.get("MODEL_BUCKET")
MODEL_BLOB_NAME = "production_model/model.joblib"
LOCAL_MODEL_PATH = "/tmp/model.joblib"
model = None

def download_model():
    print("Attempting to download the production model.")
    if not MODEL_BUCKET_NAME:
        print("Fatal Error: MODEL_BUCKET environment variable is not set.")
        return False
    print(f"Downloading from gs://{MODEL_BUCKET_NAME}/{MODEL_BLOB_NAME}.")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(MODEL_BUCKET_NAME)
        blob = bucket.blob(MODEL_BLOB_NAME)
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
        # The list of expected features now includes 'type'.
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
        # We need to create a DataFrame to hold the data in the right order.
        input_df = pd.DataFrame([data], columns=features)
        # The model was trained on numbers, so we have to encode the 'type' here too.
        # The UI will send 'red' or 'white'.
        input_df["type"] = input_df["type"].apply(lambda x: 1 if x == "red" else 0)
        print(f"Encoded input for model: {input_df.to_dict('records')}")
        # Get the prediction from our loaded model.
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