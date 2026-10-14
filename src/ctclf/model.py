"""MobileNetV3-Large classifier definition."""

from pathlib import Path

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from ctclf import config


def build_model(num_classes: int = len(config.CLASSES), pretrained: bool = True) -> nn.Module:
    """MobileNetV3-Large (ImageNet weights) with the 960->1280->num_classes head."""
    # IMAGENET1K_V1 matches torchvision's legacy `pretrained=True` used by the original repo.
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_large(weights=weights)
    model.classifier = nn.Sequential(
        nn.Linear(960, 1280),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(1280, num_classes),
    )
    return model


def save_weights(model: nn.Module, path: Path) -> None:
    """Save the model state_dict only (no optimizer)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path: Path, device: torch.device | str = "cpu") -> nn.Module:
    """Build the architecture, load weights from path and return it in eval mode."""
    model = build_model(pretrained=False)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()
