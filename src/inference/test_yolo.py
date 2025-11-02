# final_test.py
import numpy as np
import cv2
from ultralytics import YOLO
import torch

print("=== FINAL ENVIRONMENT CHECK ===")
print(f" NumPy: {np.__version__}")
print(f" OpenCV: {cv2.__version__}")
print(f" PyTorch: {torch.__version__}")
print(f" CUDA: {torch.cuda.is_available()}")
print(f" GPU: {torch.cuda.get_device_name(0)}")

print("\n=== TESTING YOLO ===")
model = YOLO('yolov8n.pt')
print(" YOLO model loaded successfully!")

# Test inference
dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
results = model(dummy_image)
print(" Inference test passed!")

print("\n YOUR CAR DAMAGE DETECTION ENVIRONMENT IS READY!")
print("   All dependencies are stable and working correctly.")