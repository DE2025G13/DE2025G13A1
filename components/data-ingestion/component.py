import argparse
import os
import traceback
from datetime import datetime
from google.cloud import storage

def upload_error_log(bucket_name: str, error_message: str):
    """Uploads a detailed error log to a specified GCS bucket."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_blob_name = f"pipeline_error_logs/data_ingestion_{timestamp}.log"
        
        blob = bucket.blob(log_blob_name)
        blob.upload_from_string(error_message)
        print(f"Successfully uploaded error log to gs://{bucket_name}/{log_blob_name}")
    except Exception as e:
        # If logging itself fails, print the original error and the logging error
        print(f"FATAL: Could not upload error log to GCS. Error: {e}")
        print("--- Original Error ---")
        print(error_message)

def ingest_data(
    bucket_name: str,
    blob_name: str,
    output_dataset_path: str,
    error_log_bucket: str,
):
    """Downloads a file from GCS into a KFP Dataset artifact directory."""
    try:
        # KFP provides a directory path for Dataset outputs. We need to create it.
        os.makedirs(output_dataset_path, exist_ok=True)
        
        # Define the full path for the output file inside the dataset directory.
        output_filename = os.path.join(output_dataset_path, "data.csv")
        
        print(f"Attempting to download gs://{bucket_name}/{blob_name} to {output_filename}")
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        print("Downloading blob...")
        blob.download_to_filename(output_filename)
        print(f"Download successful! Data saved to {output_filename}")

    except Exception:
        # Catch ANY exception that occurs
        error_message = traceback.format_exc()
        print(f"ERROR: An exception occurred in the data ingestion component. Full traceback:\n{error_message}")
        
        # Upload the detailed error log to GCS
        upload_error_log(error_log_bucket, error_message)
        
        # Re-raise the exception to ensure the pipeline step fails correctly
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket-name', type=str, required=True)
    parser.add_argument('--blob-name', type=str, required=True)
    parser.add_argument('--output-dataset-path', type=str, required=True)
    parser.add_argument('--error-log-bucket', type=str, required=True)
    args = parser.parse_args()
    ingest_data(
        args.bucket_name,
        args.blob_name,
        args.output_dataset_path,
        args.error_log_bucket
    )