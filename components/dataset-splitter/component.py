import os
import argparse
import random
import yaml
from google.cloud import storage

def create_yaml_split(bucket_name, prefix):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    blobs = storage_client.list_blobs(bucket_name, prefix=os.path.join(prefix, 'images'))
    image_files = [f"gs://{bucket_name}/{blob.name}" for blob in blobs if not blob.name.endswith('/')]
    
    random.shuffle(image_files)
    
    n = len(image_files)
    train_end = int(n * 0.7)
    val_end = train_end + int(n * 0.15)

    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]

    data_yaml_content = {
        'train': train_files,
        'val': val_files,
        'test': test_files,
        'nc': 2,
        'names': {0: 'no_glasses', 1: 'glasses'}
    }

    yaml_path = '/tmp/data.yaml'
    with open(yaml_path, 'w') as yf:
        yaml.dump(data_yaml_content, yf, default_flow_style=False)
    
    blob = bucket.blob(os.path.join(prefix, 'data.yaml'))
    blob.upload_from_filename(yaml_path)
    print(f"Uploaded data.yaml to gs://{bucket_name}/{prefix}/data.yaml")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket-name', type=str, required=True)
    parser.add_argument('--prefix', type=str, required=True)
    args = parser.parse_args()
    create_yaml_split(args.bucket_name, args.prefix)