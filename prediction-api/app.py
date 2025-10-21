from flask import Flask, request, jsonify
import os
from google.cloud import storage
import tempfile
import cv2
import numpy as np
import base64
from ultralytics import YOLO

app = Flask(__name__)

MODEL_BUCKET = os.environ.get("MODEL_BUCKET")
MODEL_BLOB = "production_model/best.pt"
_, model_file = tempfile.mkstemp(suffix=".pt")
model = None

if MODEL_BUCKET:
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(MODEL_BUCKET)
        blob = bucket.blob(MODEL_BLOB)
        blob.download_to_filename(model_file)
        model = YOLO(model_file)
        print("YOLOv8 model loaded successfully from GCS.")
    except Exception as e:
        print(f"Error loading model from GCS: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model is not loaded'}), 500
    
    file = request.files.get('file')
    if not file: return jsonify({'error': 'No file part'}), 400
        
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)
    
    result = results[0]
    if len(result.boxes) == 0:
        return jsonify({'error': 'No face detected by model'}), 400

    box = result.boxes[0]
    coords = box.xyxy[0].tolist()
    class_id = int(box.cls[0].item())
    conf = box.conf[0].item()
    label = result.names[class_id]
    
    x1, y1, x2, y2 = map(int, coords)
    color = (0, 255, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    text_label = f"{label}: {conf:.2f}"
    cv2.putText(img, text_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'prediction': label,
        'confidence': conf,
        'image_with_box': img_base64
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))