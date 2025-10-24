import argparse
from google.cloud import storage
import os

def ingest_data(bucket_name: str, blob_name: str, output_path: str):
    """Downloads a file from GCS to a local path for use as a KFP artifact."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(output_path)
    print(f"Downloaded gs://{bucket_name}/{blob_name} to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket-name', type=str, required=True)
    parser.add_argument('--blob-name', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    args = parser.parse_args()
    ingest_data(args.bucket_name, args.blob_name, args.output_path)
