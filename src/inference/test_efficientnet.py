# test_efficientnet.py
import torch
import timm
import numpy as np
import cv2

print("=== EFFICIENTNET / CAR DAMAGE CLASSIFICATION ENVIRONMENT TEST ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Test basic GPU functionality
if torch.cuda.is_available():
    x = torch.randn(1, 3, 224, 224).cuda()  # Dummy input tensor
    print(f"✓ GPU tensor allocated: {x.shape} on {x.device}")

# Load EfficientNet model from timm
model_name = "efficientnet_b0"
model = timm.create_model(model_name, pretrained=True, num_classes=3)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
print(f" EfficientNet model ({model_name}) loaded successfully!")

# Test inference with a dummy image
dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
# Convert to tensor and normalize
img_tensor = torch.tensor(dummy_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
img_tensor = img_tensor.to(device)

with torch.no_grad():
    outputs = model(img_tensor)
    predicted_class = outputs.argmax(dim=1).item()

print(f" Inference test passed! Predicted class index: {predicted_class}")
print("\nYOUR EfficientNet CAR DAMAGE CLASSIFICATION ENVIRONMENT IS READY!")
