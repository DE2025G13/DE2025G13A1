import argparse
import os
from google.cloud import storage
from google.api_core import exceptions

def ingest_data(bucket_name: str, blob_name: str, output_dataset_path: str):
    """
    Downloads a file from GCS into a KFP Dataset artifact directory.
    """
    
    os.makedirs(output_dataset_path, exist_ok=True)
    
    # Define the full path for the output file inside the dataset directory.
    output_filename = os.path.join(output_dataset_path, "data.csv")
    
    print(f"Attempting to download gs://{bucket_name}/{blob_name} to {output_filename}")

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        print("Downloading blob...")
        blob.download_to_filename(output_filename)
        print(f"Download successful! Data saved to {output_filename}")

    except exceptions.NotFound:
        print(f"FATAL: File not found at gs://{bucket_name}/{blob_name}")
        print("Please check that the bucket name and blob name are spelled correctly and that the file exists.")
        raise
        
    except exceptions.Forbidden as e:
        print(f"FATAL: Permission denied for gs://{bucket_name}/{blob_name}")
        print("Please ensure the pipeline's service account has the 'Storage Object Viewer' role on this bucket.")
        raise

    except Exception as e:
        print(f"An unexpected error occurred during download: {e}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket-name', type=str, required=True)
    parser.add_argument('--blob-name', type=str, required=True)
    parser.add_argument('--output-dataset-path', type=str, required=True)
    args = parser.parse_args()
    ingest_data(args.bucket_name, args.blob_name, args.output_dataset_path)