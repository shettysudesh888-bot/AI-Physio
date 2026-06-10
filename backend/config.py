"""
backend/config.py
-----------------
Shared configuration for model training, evaluation, and prediction.
"""

import os

# Image preprocessing parameters
IMAGE_SIZE = int(os.getenv("AI_PHYSIO_IMAGE_SIZE", "160"))

# ImageNet normalization parameters for DenseNet121
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
