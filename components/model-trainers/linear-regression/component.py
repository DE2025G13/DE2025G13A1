import argparse
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os
from google.cloud import storage

def train_linear_regression(processed_data_uri: str, output_model_uri: str):
    """
    Downloads processed data from GCS, trains a Linear Regression model,
    and uploads the model artifact back to GCS.
    """
    storage_client = storage.Client()
    
    def download_gcs_dir(uri, local_dir):
        bucket_name, prefix = uri.replace("gs://", "").split("/", 1)
        bucket = storage_client.bucket(bucket_name)
        blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
        os.makedirs(local_dir, exist_ok=True)
        for blob in blobs:
            if not blob.name.endswith('/'):
                destination_file_name = os.path.join(local_dir, os.path.basename(blob.name))
                blob.download_to_filename(destination_file_name)
                print(f"Downloaded {blob.name} to {destination_file_name}")

    # Download the processed data
    local_data_path = "/tmp/data"
    download_gcs_dir(processed_data_uri, local_data_path)

    X_train = pd.read_csv(f"{local_data_path}/x_train.csv")
    y_train = pd.read_csv(f"{local_data_path}/y_train.csv")

    model = LinearRegression()
    model.fit(X_train, y_train.values.ravel())
    print("Model training complete.")

    # Save and upload the model
    local_model_path = "/tmp/model.joblib"
    joblib.dump(model, local_model_path)
    
    bucket_name, blob_name = output_model_uri.replace("gs://", "").split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_model_path)
    print(f"Uploaded model to {output_model_uri}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_data_uri', type=str, required=True)
    parser.add_argument('--output_model_uri', type=str, required=True)
    args = parser.parse_args()
    train_linear_regression(args.processed_data_uri, args.output_model_uri)