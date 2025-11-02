# test_maskrcnn.py
import torch
print("=== MASK R-CNN / Detectron2 ENVIRONMENT TEST ===")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

try:
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.logger import setup_logger
    from detectron2 import model_zoo

    setup_logger()
    print("Detectron2 imported successfully!")

    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cuda"
    
    # Correct Mask R-CNN config file
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    
    # Correct Mask R-CNN weights URL
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)
    print("DefaultPredictor created successfully!")

    import numpy as np
    dummy_image = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
    outputs = predictor(dummy_image)
    print(f"Inference test passed! Detected {len(outputs['instances'])} objects")
    print(" Your Mask R-CNN environment is ready for use!")

except Exception as e:
    print(f" Error: {e}")
