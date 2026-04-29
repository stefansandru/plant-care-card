import os
import json
from typing import Optional
import torch
import torch.nn as nn
import torchvision

from .config import CONFIG


class Model(nn.Module):
    """
    EfficientNet-B1 wrapper for inference.

    This keeps a `.model` submodule so loading a LightningModule state_dict that
    was saved as `model.state_dict()` (with keys prefixed by `model.`) will work
    without key renaming.
    """

    def __init__(self, num_classes: Optional[int] = None):
        super().__init__()

        # Determine number of classes from class_map if not provided
        class_map_path = CONFIG.get("CLASS_MAP_PATH")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        class_map_full = os.path.normpath(os.path.join(base_dir, class_map_path))
        with open(class_map_full, "r") as f:
            class_map = json.load(f)
        num_classes = len(class_map)

        # Build EfficientNet-B1 backbone without pretrained weights (we'll load ours)
        self.model = torchvision.models.efficientnet_b1(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
