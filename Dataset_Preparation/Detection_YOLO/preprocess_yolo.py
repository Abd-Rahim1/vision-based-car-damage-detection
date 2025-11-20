import os
import cv2
import argparse
from pathlib import Path
import yaml

def resize_and_normalize(img, size, normalize):
    """Resize image to (size x size) and optionally normalize to [0,1]."""
    img_resized = cv2.resize(img, (size, size))
    if normalize:
        img_resized = img_resized / 255.0
    return img_resized

def preprocess_yolo(input_folder, params):
    size = params["image_size"]
    normalize = params["normalize"]

    input_folder = Path(input_folder).resolve()

    processed_count = 0
    for root, _, files in os.walk(input_folder):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = Path(root) / f
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img_processed = resize_and_normalize(img, size, normalize)
                # Save back as 0-255
                cv2.imwrite(str(img_path), (img_processed * 255).astype('uint8'))
                processed_count += 1

    print(f"YOLO preprocessing complete! Processed {processed_count} images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to folder with images")
    args = parser.parse_args()

    # Load YOLO parameters from params.yaml
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
    yolo_params = config["data_preparation"]["yolo"]

    preprocess_yolo(args.input, yolo_params)
