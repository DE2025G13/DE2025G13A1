import argparse
from ultralytics import YOLO
from google.cloud import storage

def train_detector(data_yaml_uri, output_bucket_name, output_blob_name):
    print(f"Starting object detection training with data from: {data_yaml_uri}")
    
    model = YOLO('yolov8n.pt') 
    
    results = model.train(
        data=data_yaml_uri,
        epochs=50,
        imgsz=640,
        project='/app/runs',
        name='glasses_detector_training'
    )

    best_model_path = results.save_dir / 'weights' / 'best.pt'
    print(f"Training complete. Best model is at {best_model_path}")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(output_bucket_name)
    blob = bucket.blob(output_blob_name)
    
    blob.upload_from_filename(best_model_path)
    print(f"Uploaded best detector model to gs://{output_bucket_name}/{output_blob_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-yaml-uri', type=str, required=True)
    parser.add_argument('--output-bucket-name', type=str, required=True)
    parser.add_argument('--output-blob-name', type=str, required=True)
    args = parser.parse_args()
    train_detector(args.data_yaml_uri, args.output_bucket_name, args.output_blob_name)