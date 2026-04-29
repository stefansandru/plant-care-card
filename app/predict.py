#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from .config import CONFIG


def _build_transforms() -> transforms.Compose:
    """
    Build preprocessing transforms exactly matching training notebook.
    
    Expects PIL Image input.
    ToTensor() converts PIL [0,255] uint8 -> [0,1] float32 tensor automatically.
    """
    img_size = CONFIG.get("IMG_SIZE", (32, 32))
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    mean = CONFIG.get("NORMALIZE_MEAN", [0.485, 0.456, 0.406])
    std = CONFIG.get("NORMALIZE_STD", [0.229, 0.224, 0.225])
    
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),  # CRITICAL: PIL [0,255] uint8 -> [0,1] float32 tensor
        transforms.Normalize(mean=mean, std=std)
    ])


def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    """
    Prepare a PIL Image for EfficientNet inference.
    
    Matches notebook preprocessing exactly:
    - Resize to (32, 32)
    - ToTensor(): PIL [0,255] -> torch [0,1] 
    - Normalize with ImageNet stats
    
    Returns batched tensor [1, 3, H, W].
    """
    tfm = _build_transforms()
    tensor = tfm(pil_image)  # [3, H, W]
    
    # Add batch dimension
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)  # [1, 3, H, W]
    
    return tensor


def predict(package: dict, pil_image: Image.Image) -> np.ndarray:
    """
    Run model and get softmax probabilities for a single PIL Image.

    :param package: dict from fastapi state including 'model'
    :param pil_image: PIL.Image.Image in RGB mode
    :return: numpy array of shape [num_classes] with probabilities
    """
    model = package["model"]
    model.eval()

    # preprocess to batch tensor (exactly as in notebook)
    X = preprocess_image(pil_image)

    with torch.no_grad():
        X = X.to(CONFIG["DEVICE"])  # [1, 3, H, W]
        logits = model(X)            # [1, num_classes]
        probs = torch.softmax(logits, dim=1)

    return probs.squeeze(0).cpu().numpy()
