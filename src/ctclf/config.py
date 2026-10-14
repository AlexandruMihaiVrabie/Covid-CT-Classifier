"""Paths, hyperparameters and device selection."""

import os
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset root (read-only); override with the CTCLF_DATA_ROOT env var.
DATA_ROOT = Path(os.environ.get("CTCLF_DATA_ROOT", PROJECT_ROOT / "archive" / "Original CT Scans"))

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
RESULTS_DIR = OUTPUTS_DIR / "results"

# Class order defines the label index: nCT=0, pCT=1, NiCT=2.
CLASSES: tuple[str, ...] = ("nCT", "pCT", "NiCT")
CLASS_DESCRIPTIONS: dict[str, str] = {
    "nCT": "normal",
    "pCT": "COVID-19 positive",
    "NiCT": "non-informative",
}
IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png")

# Split: per class, fixed counts, deterministic.
SEED = 42
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15

# Preprocessing (ImageNet statistics).
IMAGE_SIZE = 224
RESIZE_SIZE = 256
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# Training defaults (sized for an 8 GB M1).
BATCH_SIZE = 16
NUM_WORKERS = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4


def get_device() -> torch.device:
    """Return MPS if available, otherwise CPU."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
