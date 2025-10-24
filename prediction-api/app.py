from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
from google.cloud import storage

app = Flask(__name__)

MODEL_BUCKET_NAME = os.environ.get("MODEL_BUCKET", "yannick-wine-models")
MODEL_BLOB_NAME = "production_model/model.joblib"
LOCAL_MODEL_PATH = "/tmp/model.joblib"

model = None

def download_model():
    """Downloads the model file from GCS."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(MODEL_BUCKET_NAME)
        blob = bucket.blob(MODEL_BLOB_NAME)
        blob.download_to_filename(LOCAL_MODEL_PATH)
        print(f"Model downloaded from gs://{MODEL_BUCKET_NAME}/{MODEL_BLOB_NAME}")
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def load_model():
    """Loads the model from the local file into memory."""
    global model
    if os.path.exists(LOCAL_MODEL_PATH):
        try:
            model = joblib.load(LOCAL_MODEL_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model from disk: {e}")
            model = None
    else:
        print("Model file not found locally.")
        model = None

if download_model():
    load_model()

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not available. Check server logs.'}), 503

    try:
        data = request.get_json()
        features = [
            "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
            "pH", "sulphates", "alcohol"
        ]
        
        input_df = pd.DataFrame([data], columns=features)
        
        prediction_result = model.predict(input_df)
        
        final_prediction = int(round(prediction_result[0]))

        return jsonify({'prediction': final_prediction})

    except Exception as e:
        return jsonify({'error': f"An error occurred during prediction: {e}"}), 400

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    if model:
        return jsonify({'status': 'ok', 'model': 'loaded'}), 200
    else:
        return jsonify({'status': 'error', 'model': 'not_loaded'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))