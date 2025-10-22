import argparse
import logging
import sys
import traceback
from ultralytics import YOLO
from google.cloud import storage
from datetime import datetime

def log_fatal_error_to_gcs(exc_info, output_bucket_name):
    """Logs a fatal traceback to a file in GCS for foolproof debugging."""
    try:
        error_string = "".join(traceback.format_exception(*exc_info))
        
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        error_log_path = f"component_errors/yolo-trainer/{timestamp}-error.log"
        
        client = storage.Client()
        bucket = client.bucket(output_bucket_name)
        blob = bucket.blob(error_log_path)
        blob.upload_from_string(
            f"yolo-trainer component failed with a fatal error:\n\n{error_string}",
            content_type="text/plain",
        )
        print(f"A fatal error was logged to gs://{output_bucket_name}/{error_log_path}", file=sys.stderr)
    except Exception as e:
        print(f"CRITICAL: Could not log fatal error to GCS. GCS Error: {e}", file=sys.stderr)
        print(f"ORIGINAL FATAL ERROR: {''.join(traceback.format_exception(*exc_info))}", file=sys.stderr)

def train_detector(data_yaml_uri, output_bucket_name, output_blob_name):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
    
    logging.info("--- Starting YOLOv8 Training Task ---")
    logging.info(f"Using data configuration from: {data_yaml_uri}")

    model = YOLO('yolov8n.pt')

    logging.info("Starting model training...")
    results = model.train(
        data=data_yaml_uri,
        epochs=3,
        imgsz=640,
        project='/app/runs',
        name='glasses_detector_training'
    )

    best_model_path = results.save_dir / 'weights' / 'best.pt'
    logging.info(f"Training complete. Best model is at local path: {best_model_path}")

    logging.info("Uploading best model to Google Cloud Storage...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(output_bucket_name)
    blob = bucket.blob(output_blob_name)
    blob.upload_from_filename(best_model_path)
    logging.info(f"Successfully uploaded model to gs://{output_bucket_name}/{output_blob_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-yaml-uri', type=str, required=True)
    parser.add_argument('--output-bucket-name', type=str, required=True)
    parser.add_argument('--output-blob-name', type=str, required=True)
    args = parser.parse_args()

    try:
        train_detector(args.data_yaml_uri, args.output_bucket_name, args.output_blob_name)
    except Exception:
        log_fatal_error_to_gcs(sys.exc_info(), args.output_bucket_name)
        sys.exit(1)