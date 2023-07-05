import os
import cv2
from pathlib import Path

# ---------- config parameters from json ----------
OPENCV_CONFIG = {
    'image_multiplier': 0.00392,  # 1 / 255
    'image_spatial_size': (1920, 1088),
    'fp16': True,  # inference
    'image_mean_subtraction': [127.5, 127.5, 127.5],
    'blob_normalization_scale': 1.0,
    'blob_mean_subtraction': [0, 0, 0],
    'blob_standard_deviation': [0.5, 0.5, 0.5],
    'swapRB': True,
    'crop': None,
    'ddepth': cv2.CV_32F,
}
# ========== /config parameters from json ==========
N_CPU = os.cpu_count()

MODELS_DIR = Path('/home/vid/hdd/file/project/143-NLMK-DCA/Theme4Dim/models/')
PROJECT_DIR = Path('/home/vid/hdd/projects/PycharmProjects/segmentation_models.pytorch-0.1.3')
