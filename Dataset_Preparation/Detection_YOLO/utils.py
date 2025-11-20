import os
import json
import yaml
import cv2
import numpy as np
from PIL import Image
import random
import argparse
from typing import Dict, List, Tuple, Any

def load_config(config_path: str = "params.yaml") -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_directory(path: str):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """Get image dimensions without loading full image"""
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception as e:
        return (0, 0)

def coco_to_yolo_bbox(bbox: List[float], img_width: int, img_height: int) -> List[float]:
    """Convert COCO bbox format to YOLO format"""
    x, y, w, h = bbox
    
    # YOLO format: normalized center_x, center_y, width, height
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    
    return [x_center, y_center, w_norm, h_norm]

def calculate_bbox_area(bbox: List[float]) -> float:
    """Calculate area of bounding box"""
    x, y, w, h = bbox
    return w * h

def calculate_bbox_aspect_ratio(bbox: List[float]) -> float:
    """Calculate aspect ratio of bounding box"""
    x, y, w, h = bbox
    return max(w, h) / min(w, h) if min(w, h) > 0 else float('inf')

def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate Intersection over Union for two bounding boxes"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Convert to [x1, y1, x2, y2] format
    box1 = [x1, y1, x1 + w1, y1 + h1]
    box2 = [x2, y2, x2 + w2, y2 + h2]
    
    # Calculate intersection area
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize image to target size"""
    return cv2.resize(image, target_size)

def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image pixel values to 0-1"""
    return image.astype(np.float32) / 255.0

def save_image(image: np.ndarray, output_path: str):
    """Save image to file"""
    create_directory(os.path.dirname(output_path))
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def load_coco_annotations(coco_path: str, split: str) -> Dict:
    """Load COCO annotations for a specific split"""
    ann_file = os.path.join(coco_path, "annotations", f"instances_{split}.json")
    if not os.path.exists(ann_file):
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")
    
    with open(ann_file, 'r') as f:
        return json.load(f)

def save_coco_annotations(coco_data: Dict, output_path: str, split: str):
    """Save COCO annotations to file"""
    ann_file = os.path.join(output_path, "annotations", f"instances_{split}.json")
    create_directory(os.path.dirname(ann_file))
    with open(ann_file, 'w') as f:
        json.dump(coco_data, f, indent=2)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Dataset Preparation Pipeline')
    parser.add_argument('--input', type=str, help='Input dataset path')
    parser.add_argument('--output', type=str, help='Output dataset path')
    parser.add_argument('--config', type=str, default='params.yaml', help='Config file path')
    return parser.parse_args()