import os
import argparse
import glob
import logging
import cv2
import mediapipe as mp
from google.cloud import storage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def generate_labels(raw_bucket_name, raw_prefix, processed_bucket_name, processed_prefix):
    storage_client = storage.Client()
    raw_bucket = storage_client.bucket(raw_bucket_name)
    processed_bucket = storage_client.bucket(processed_bucket_name)

    local_raw_dir = "/tmp/raw_data"
    os.makedirs(local_raw_dir, exist_ok=True)

    logging.info(f"Downloading images from gs://{raw_bucket_name}/{raw_prefix}")
    blobs = storage_client.list_blobs(raw_bucket_name, prefix=raw_prefix)
    for blob in blobs:
        if not blob.name.endswith('/'):
            destination_folder = os.path.join(local_raw_dir, os.path.dirname(blob.name))
            os.makedirs(destination_folder, exist_ok=True)
            blob.download_to_filename(os.path.join(local_raw_dir, blob.name))

    logging.info("Starting label generation...")
    classes = ["no_glasses", "glasses"]
    for cls_name in classes:
        cls_id = classes.index(cls_name)
        image_paths = glob.glob(os.path.join(local_raw_dir, raw_prefix, cls_name, "*"))
        
        for img_path in image_paths:
            fname = os.path.basename(img_path)
            try:
                img = cv2.imread(img_path)
                if img is None: raise ValueError("Image could not be read.")
                h, w = img.shape[:2]

                results = face_detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if not results.detections: raise ValueError("No face detected.")

                det = results.detections[0].location_data.relative_bounding_box
                x_center, y_center = det.xmin + det.width / 2, det.ymin + det.height / 2
                box_w, box_h = det.width, det.height

                label_fname = os.path.splitext(fname)[0] + ".txt"
                local_label_path = f"/tmp/{label_fname}"
                with open(local_label_path, "w") as f:
                    f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")
                
                img_gcs_path = os.path.join(processed_prefix, 'images', fname)
                processed_bucket.blob(img_gcs_path).upload_from_filename(img_path)

                label_gcs_path = os.path.join(processed_prefix, 'labels', label_fname)
                processed_bucket.blob(label_gcs_path).upload_from_filename(local_label_path)
                
                logging.info(f"Uploaded {fname} and its label.")

            except Exception as e:
                logging.warning(f"Skipping {fname}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-bucket-name', type=str, required=True)
    parser.add_argument('--raw-prefix', type=str, default='raw_images')
    parser.add_argument('--processed-bucket-name', type=str, required=True)
    parser.add_argument('--processed-prefix', type=str, required=True)
    args = parser.parse_args()
    generate_labels(args.raw_bucket_name, args.raw_prefix, args.processed_bucket_name, args.processed_prefix)