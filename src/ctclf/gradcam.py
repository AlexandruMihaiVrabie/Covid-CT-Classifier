"""Grad-CAM heatmaps and overlays."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures are only written to disk
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from pytorch_grad_cam import GradCAM  # noqa: E402
from pytorch_grad_cam.utils.image import show_cam_on_image  # noqa: E402
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget  # noqa: E402
from torch import nn  # noqa: E402
from torchvision import transforms as T  # noqa: E402

from ctclf import config  # noqa: E402
from ctclf.data import build_index, load_image, split_index  # noqa: E402
from ctclf.model import load_model  # noqa: E402
from ctclf.predict import DEFAULT_CHECKPOINT, predict_image, preprocess  # noqa: E402

ASSETS_DIR = config.PROJECT_ROOT / "assets"
EXAMPLES_NAME = "gradcam_examples.png"


def get_target_layer(model: nn.Module) -> nn.Module:
    """Last convolutional block of the MobileNetV3 backbone (160 -> 960 channels)."""
    return model.features[-1]


def display_image(image: Image.Image) -> np.ndarray:
    """The 224x224 crop the model sees, as float RGB in [0, 1] (no normalization)."""
    crop = T.Compose([T.Resize(config.RESIZE_SIZE), T.CenterCrop(config.IMAGE_SIZE)])
    return np.asarray(crop(image.convert("RGB")), dtype=np.float32) / 255.0


def compute_gradcam(
    model: nn.Module,
    image: Image.Image,
    target_class: int | None = None,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, int]:
    """Grad-CAM heatmap in [0, 1] at input size and the class it explains (default: predicted)."""
    model.eval()
    input_tensor = preprocess(image).to(device)
    if target_class is None:
        with torch.no_grad():
            target_class = int(model(input_tensor).argmax(1).item())
    with GradCAM(model=model, target_layers=[get_target_layer(model)]) as cam:
        heatmap = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_class)])
    return heatmap[0], target_class


def overlay(image: Image.Image, heatmap: np.ndarray, image_weight: float = 0.5) -> np.ndarray:
    """Blend the heatmap (jet colormap) onto the model's view of the slice; uint8 RGB."""
    return show_cam_on_image(display_image(image), heatmap, use_rgb=True, image_weight=image_weight)


def explain_image(
    model: nn.Module, image: Image.Image, device: torch.device | str = "cpu"
) -> tuple[dict, np.ndarray]:
    """Prediction plus a Grad-CAM overlay for the predicted class."""
    prediction = predict_image(model, image, device)
    heatmap, _ = compute_gradcam(model, image, prediction["label_index"], device)
    return prediction, overlay(image, heatmap)


def pick_examples(per_class: int = 1) -> list[Path]:
    """First test-split slices of each class (deterministic, from the seed-42 split)."""
    test = split_index(build_index())["test"]
    picked: list[Path] = []
    for label in range(len(config.CLASSES)):
        picked.extend([s.path for s in test if s.label == label][:per_class])
    return picked


def save_examples(
    model: nn.Module, paths: list[Path], out_dir: Path, device: torch.device | str = "cpu"
) -> list[Path]:
    """Save one overlay per image plus a grid figure (slice | overlay) to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(paths), 2, figsize=(6, 3 * len(paths)), squeeze=False)
    written: list[Path] = []
    for row, path in zip(axes, paths, strict=True):
        image = load_image(path)
        prediction, blended = explain_image(model, image, device)
        true_cls = path.parent.name
        out = out_dir / f"gradcam_{true_cls}.png"
        Image.fromarray(blended).save(out)
        written.append(out)
        p = prediction["probabilities"][prediction["label"]]
        row[0].imshow(display_image(image))
        row[0].set_title(f"{path.name} (true: {true_cls})", fontsize=9)
        row[1].imshow(blended)
        row[1].set_title(f"pred: {prediction['label']} p={p:.3f}", fontsize=9)
        for ax in row:
            ax.axis("off")
        print(f"{path.name}: true {true_cls}, pred {prediction['label']} p={p:.4f} -> {out.name}")
    fig.suptitle("Grad-CAM (educational, not a medical device)", fontsize=10)
    fig.tight_layout()
    grid = out_dir / EXAMPLES_NAME
    fig.savefig(grid, dpi=120)
    plt.close(fig)
    written.append(grid)
    return written


def main() -> None:
    """Save a Grad-CAM overlay for one image, or example overlays for each class."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("image", type=Path, nargs="?", help="omit with --examples")
    parser.add_argument("--out", type=Path, help="output PNG (default: <image stem>_gradcam.png)")
    parser.add_argument("--examples", action="store_true", help="one test slice per class")
    parser.add_argument("--out-dir", type=Path, default=ASSETS_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", help="override auto-detection, e.g. cpu")
    args = parser.parse_args()
    if not args.examples and args.image is None:
        parser.error("give an image path or --examples")

    device = torch.device(args.device) if args.device else config.get_device()
    model = load_model(args.checkpoint, device)
    if args.examples:
        written = save_examples(model, pick_examples(), args.out_dir, device)
        print("Wrote " + ", ".join(str(p) for p in written))
        return

    prediction, blended = explain_image(model, load_image(args.image), device)
    out = args.out or Path(f"{args.image.stem}_gradcam.png")
    Image.fromarray(blended).save(out)
    print(f"{args.image.name}: {prediction['label']} ({prediction['description']}) -> {out}")


if __name__ == "__main__":
    main()
