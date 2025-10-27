import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from google.cloud import storage

def split_data(input_data_gcs_path: str, output_dataset_path: str):
    print(f"Downloading data from GCS: {input_data_gcs_path}")
    gcs_path_parts = input_data_gcs_path.replace("gs://", "").split("/", 1)
    bucket_name = gcs_path_parts[0]
    blob_path = gcs_path_parts[1]
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    local_file = "/tmp/wine.csv"
    blob.download_to_filename(local_file)
    print(f"Downloaded to {local_file}")
    df = pd.read_csv(local_file, sep=";")
    if "Id" in df.columns:
        print("Found and removed the 'Id' column.")
        df.drop(columns="Id", inplace=True)
    print("Encoding the 'type' column (white=0, red=1).")
    df["type"] = df["type"].apply(lambda x: 1 if x == "red" else 0)
    print("Separating features (X) from the target (y).")
    X = df.drop(columns=["quality"])
    y = df["quality"]
    print("Performing an 80/20 train-test split.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    os.makedirs(output_dataset_path, exist_ok=True)
    print("Saving split datasets to output paths.")
    X_train.to_csv(os.path.join(output_dataset_path, "x_train.csv"), index=False)
    y_train.to_csv(os.path.join(output_dataset_path, "y_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dataset_path, "x_test.csv"), index=False)
    y_test.to_csv(os.path.join(output_dataset_path, "y_test.csv"), index=False)
    print("Data splitting and saving completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data-gcs-path", type=str, required=True)
    parser.add_argument("--output-dataset-path", type=str, required=True)
    args = parser.parse_args()
    split_data(args.input_data_gcs_path, args.output_dataset_path)