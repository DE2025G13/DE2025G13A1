import argparse
import os
import pandas as pd
from google.cloud import storage
import requests

def load_data(input_data_gcs_path: str, output_dataset_path: str):
    local_file = "/tmp/wine.csv"
    
    if input_data_gcs_path.startswith("http://") or input_data_gcs_path.startswith("https://"):
        print(f"Downloading data from URL: {input_data_gcs_path}")
        response = requests.get(input_data_gcs_path)
        response.raise_for_status()
        with open(local_file, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded to {local_file}")
    elif input_data_gcs_path.startswith("gs://"):
        print(f"Downloading data from GCS: {input_data_gcs_path}")
        gcs_path_parts = input_data_gcs_path.replace("gs://", "").split("/", 1)
        bucket_name = gcs_path_parts[0]
        blob_path = gcs_path_parts[1]
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.download_to_filename(local_file)
        print(f"Downloaded to {local_file}")
    else:
        raise ValueError(f"Unsupported input source: {input_data_gcs_path}")
    
    df = pd.read_csv(local_file, sep=";")
    if "Id" in df.columns:
        print("Found and removed the 'Id' column.")
        df.drop(columns="Id", inplace=True)
    print("Encoding the 'type' column (white=0, red=1).")
    df["type"] = df["type"].apply(lambda x: 1 if x == "red" else 0)
    os.makedirs(output_dataset_path, exist_ok=True)
    output_file = os.path.join(output_dataset_path, "wine.csv")
    df.to_csv(output_file, index=False)
    print(f"Saved preprocessed dataset to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data-gcs-path", type=str, required=True)
    parser.add_argument("--output-dataset-path", type=str, required=True)
    args = parser.parse_args()
    load_data(args.input_data_gcs_path, args.output_dataset_path)