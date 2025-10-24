import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from google.cloud import storage

def preprocess_data(data_bucket_name: str, raw_data_path: str, processed_prefix: str):
    storage_client = storage.Client()
    bucket = storage_client.bucket(data_bucket_name)

    # Download raw data
    local_raw_path = "/tmp/winequality.csv"
    bucket.blob(raw_data_path).download_to_filename(local_raw_path)

    df = pd.read_csv(local_raw_path)
    df.drop(columns="Id", inplace=True)

    X = df.drop(columns=["quality"])
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save files locally first
    local_dir = "/tmp/processed"
    os.makedirs(local_dir, exist_ok=True)
    X_train.to_csv(f"{local_dir}/x_train.csv", index=False)
    y_train.to_csv(f"{local_dir}/y_train.csv", index=False)
    X_test.to_csv(f"{local_dir}/x_test.csv", index=False)
    y_test.to_csv(f"{local_dir}/y_test.csv", index=False)

    # Upload each processed file to GCS
    for filename in os.listdir(local_dir):
        blob_path = os.path.join(processed_prefix, filename)
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(os.path.join(local_dir, filename))
        print(f"Uploaded {filename} to gs://{data_bucket_name}/{blob_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-bucket-name', type=str, required=True)
    parser.add_argument('--raw-data-path', type=str, required=True)
    parser.add_argument('--processed-prefix', type=str, required=True)
    args = parser.parse_args()
    preprocess_data(args.data_bucket_name, args.raw_data_path, args.processed_prefix)