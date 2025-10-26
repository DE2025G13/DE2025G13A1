import argparse
import os
from google.cloud import storage

def ingest_data(bucket_name: str, blob_name: str, output_dataset_path: str):
    """Downloads a file from GCS into the KFP-provided output directory."""

    os.makedirs(output_dataset_path, exist_ok=True)
    output_filename = os.path.join(output_dataset_path, "data.csv")
    
    print(f"Attempting to download gs://{bucket_name}/{blob_name} to {output_filename}")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    blob.download_to_filename(output_filename)
    print(f"Download successful! Data saved to {output_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket-name', type=str, required=True, help='GCS bucket where the raw data is stored.')
    parser.add_argument('--blob-name', type=str, required=True, help='Path to the raw data file within the GCS bucket.')
    parser.add_argument('--output-dataset-path', type=str, required=True, help='Path provided by KFP to store the output Dataset artifact.')
    args = parser.parse_args()
    
    ingest_data(
        args.bucket_name,
        args.blob_name,
        args.output_dataset_path
    )