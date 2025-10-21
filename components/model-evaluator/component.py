import argparse
import os
from ultralytics import YOLO
from google.cloud import storage

def evaluate_detectors(data_yaml_uri, new_model_uri, prod_model_uri):
    storage_client = storage.Client()
    
    def download_blob(uri, destination_file_name):
        bucket_name, blob_name = uri.replace("gs://", "").split("/", 1)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(destination_file_name)
        print(f"Downloaded {uri} to {destination_file_name}")

    os.makedirs('/app/models', exist_ok=True)
    local_new_model = '/app/models/new_model.pt'
    local_prod_model = '/app/models/prod_model.pt'
    
    download_blob(new_model_uri, local_new_model)
    new_model = YOLO(local_new_model)
    
    try:
        download_blob(prod_model_uri, local_prod_model)
        prod_model = YOLO(local_prod_model)
    except Exception as e:
        print(f"Could not load production model. Assuming new model is better. Error: {e}")
        prod_model = None

    print("Evaluating new model...")
    new_metrics = new_model.val(data=data_yaml_uri, split='test')
    new_map = new_metrics.box.map
    print(f"New model mAP50-95: {new_map}")

    if prod_model:
        print("Evaluating production model...")
        prod_metrics = prod_model.val(data=data_yaml_uri, split='test')
        prod_map = prod_metrics.box.map
        print(f"Production model mAP50-95: {prod_map}")
        
        if new_map > prod_map:
            print("deploy")
        else:
            print("keep_old")
    else:
        print("deploy")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-yaml-uri', type=str, required=True)
    parser.add_argument('--new-model-uri', type=str, required=True)
    parser.add_argument('--prod-model-uri', type=str, required=True)
    args = parser.parse_args()
    evaluate_detectors(args.data_yaml_uri, args.new_model_uri, args.prod_model_uri)