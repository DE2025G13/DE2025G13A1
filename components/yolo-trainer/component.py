import argparse
import logging
import sys
import traceback
from ultralytics import YOLO
from google.cloud import storage
from pathlib import Path

def download_gcs_file(gcs_uri: str, local_path: str):
    """Downloads a file from GCS to a local path."""
    logging.info(f"Downloading {gcs_uri} to {local_path}...")
    storage_client = storage.Client()
    bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(local_path)
    logging.info("Download complete.")

def train_detector(data_yaml_uri, output_bucket_name, output_blob_name):
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            stream=sys.stdout
        )

        logging.info("--- Starting YOLOv8 Training Task ---")
        
        local_data_yaml_path = "/tmp/data.yaml"
        download_gcs_file(data_yaml_uri, local_data_yaml_path)

        logging.info("Loading pre-trained model: yolov8n.pt")
        model = YOLO('yolov8n.pt')

        logging.info("Starting model training...")
        results = model.train(
            data=local_data_yaml_path,
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
        logging.info("--- YOLOv8 Training Task Finished Successfully ---")

    except Exception as e:
        logging.error(f"An error occurred during the training task: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-yaml-uri', type=str, required=True)
    parser.add_argument('--output-bucket-name', type=str, required=True)
    parser.add_argument('--output-blob-name', type=str, required=True)
    args = parser.parse_args()
    train_detector(args.data_yaml_uri, args.output_bucket_name, args.output_blob_name)