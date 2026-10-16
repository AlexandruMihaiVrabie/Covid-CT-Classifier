"""Single-image inference CLI."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor, nn

from ctclf import config
from ctclf.data import get_transforms, load_image
from ctclf.model import load_model
from ctclf.train import BEST_NAME

DEFAULT_CHECKPOINT = config.CHECKPOINTS_DIR / BEST_NAME


def preprocess(image: Image.Image) -> Tensor:
    """Eval transform (resize 256, center crop 224, normalize) as a (1, 3, 224, 224) batch."""
    return get_transforms(train=False)(image.convert("RGB")).unsqueeze(0)


@torch.no_grad()
def predict_image(model: nn.Module, image: Image.Image, device: torch.device | str = "cpu") -> dict:
    """Predicted class, its description and softmax probabilities for one image."""
    model.eval()
    logits = model(preprocess(image).to(device))
    probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
    idx = max(range(len(probs)), key=probs.__getitem__)
    label = config.CLASSES[idx]
    return {
        "label": label,
        "label_index": idx,
        "description": config.CLASS_DESCRIPTIONS[label],
        "probabilities": dict(zip(config.CLASSES, probs, strict=True)),
    }


def predict_path(
    path: Path, checkpoint: Path = DEFAULT_CHECKPOINT, device: torch.device | str = "cpu"
) -> dict:
    """Load the checkpoint and classify the image at path."""
    model = load_model(checkpoint, device)
    return {"image": str(path), **predict_image(model, load_image(path), device)}


def main() -> None:
    """Classify one CT slice and print the class and probabilities."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", help="override auto-detection, e.g. cpu")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else config.get_device()
    result = predict_path(args.image, args.checkpoint, device)
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"{args.image.name}: {result['label']} ({result['description']})")
    for cls, p in result["probabilities"].items():
        print(f"  {cls:5s} {p:.4f}")
    print("Educational use only, not a medical device.")


if __name__ == "__main__":
    main()
