import argparse
import logging
import sys
import traceback
from ultralytics import YOLO
from google.cloud import storage

def log_fatal_error(exc_info, output_bucket_name):
    """Logs a fatal error to a GCS file for debugging."""
    try:
        error_string = "".join(traceback.format_exception(*exc_info))
        
        error_log_path = f"component_errors/yolo-trainer/{job_id_or_timestamp}/error.log"
        
        client = storage.Client()
        bucket = client.bucket(output_bucket_name)
        blob = bucket.blob(error_log_path)
        blob.upload_from_string(
            f"yolo-trainer component failed with a fatal error:\n\n{error_string}",
            content_type="text/plain",
        )
        print(f"FATAL ERROR: {error_string}", file=sys.stderr)
    except Exception as e:
        print(f"CRITICAL: Failed to log fatal error to GCS: {e}", file=sys.stderr)
        print(f"ORIGINAL FATAL ERROR: {exc_info}", file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-yaml-uri', type=str, required=True)
    parser.add_argument('--output-bucket-name', type=str, required=True)
    parser.add_argument('--output-blob-name', type=str, required=True)
    args = parser.parse_args()

    from datetime import datetime
    job_id_or_timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')

    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            stream=sys.stdout
        )

        logging.info("--- Starting YOLOv8 Training Task ---")
        logging.info(f"Using data configuration from: {args.data_yaml_uri}")
        logging.info(f"Output model will be saved to: gs://{args.output_bucket_name}/{args.output_blob_name}")

        logging.info("Loading pre-trained model: yolov8n.pt")
        model = YOLO('yolov8n.pt')

        logging.info("Starting model training...")
        results = model.train(
            data=args.data_yaml_uri,
            epochs=3,
            imgsz=640,
            project='/app/runs',
            name='glasses_detector_training'
        )

        best_model_path = results.save_dir / 'weights' / 'best.pt'
        logging.info(f"Training complete. Best model is located at local path: {best_model_path}")

        logging.info("Uploading best model to Google Cloud Storage...")
        storage_client = storage.Client()
        bucket = storage_client.bucket(args.output_bucket_name)
        blob = bucket.blob(args.output_blob_name)

        blob.upload_from_filename(best_model_path)
        logging.info(f"Successfully uploaded model to gs://{args.output_bucket_name}/{args.output_blob_name}")
        logging.info("--- YOLOv8 Training Task Finished Successfully ---")
        
    except Exception:
        log_fatal_error(sys.exc_info(), args.output_bucket_name)
        sys.exit(1)