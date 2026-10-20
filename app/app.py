"""Local Gradio UI: CNN prediction and Grad-CAM for one chest-CT slice (no network, no API)."""

import os

# Set before importing gradio so no usage analytics are sent.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from ctclf import config  # noqa: E402
from ctclf.gradcam import explain_image  # noqa: E402
from ctclf.model import load_model  # noqa: E402
from ctclf.predict import DEFAULT_CHECKPOINT  # noqa: E402

HOST = "127.0.0.1"
PORT = 7860
EXAMPLES_DIR = config.PROJECT_ROOT / "assets" / "examples"
EXAMPLES = ("nCT7356.jpg", "pCT2369.jpg", "NiCT219.jpg")  # one test-split slice per class

DISCLAIMER = (
    "**Educational project only, not a medical device.** Do not use it for diagnosis or any "
    "clinical decision."
)
ALWAYS_A_CLASS_NOTE = (
    "The model always answers with one of its three classes (nCT normal, pCT COVID-19 "
    "positive, NiCT non-informative), even for images that are not chest-CT slices. "
    "A confident answer on such an image is meaningless."
)
ABOUT = (
    "MobileNetV3-Large fine-tuned on single chest-CT slices (COVID-CT dataset, two Wuhan "
    "hospitals; per-slice split, so test accuracy is optimistic). Grad-CAM highlights the "
    "regions that most influenced the predicted class; it is not a lesion map."
)


def summary_markdown(prediction: dict) -> str:
    """One-line verdict for the predicted class."""
    label = prediction["label"]
    p = prediction["probabilities"][label]
    return f"**Prediction: {label}** ({prediction['description']}), probability {p:.1%}"


def make_predict_fn(model: torch.nn.Module, device: torch.device | str = "cpu"):
    """Gradio callback: image -> (class probabilities, Grad-CAM overlay, summary text)."""

    def predict(image: Image.Image | None) -> tuple[dict[str, float], np.ndarray, str]:
        if image is None:
            raise gr.Error("Upload a chest-CT slice (JPG or PNG) first.")
        prediction, blended = explain_image(model, image.convert("RGB"), device)
        return prediction["probabilities"], blended, summary_markdown(prediction)

    return predict


def build_app(
    checkpoint: Path = DEFAULT_CHECKPOINT, device: torch.device | str = "cpu"
) -> gr.Blocks:
    """Load the CNN and assemble the UI."""
    predict = make_predict_fn(load_model(checkpoint, device), device)
    examples = [str(EXAMPLES_DIR / name) for name in EXAMPLES if (EXAMPLES_DIR / name).exists()]

    with gr.Blocks(title="COVID-19 CT slice classifier") as demo:
        gr.Markdown("# COVID-19 chest-CT slice classifier (CNN + Grad-CAM)")
        gr.Markdown(DISCLAIMER)
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(
                    type="pil", image_mode="RGB", sources=["upload", "clipboard"], label="CT slice"
                )
                button = gr.Button("Classify", variant="primary")
                if examples:
                    gr.Examples(examples=examples, inputs=image_in, label="Test-split examples")
            with gr.Column():
                summary = gr.Markdown()
                probs = gr.Label(num_top_classes=len(config.CLASSES), label="Class probabilities")
                cam = gr.Image(type="numpy", label="Grad-CAM (predicted class)")
        gr.Markdown(ALWAYS_A_CLASS_NOTE)
        gr.Markdown(ABOUT)
        button.click(predict, inputs=image_in, outputs=[probs, cam, summary], api_name="predict")
    return demo


def main() -> None:
    """Launch the UI on localhost only (no public share link)."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cpu", help="cpu (default) or mps")
    args = parser.parse_args()
    demo = build_app(args.checkpoint, torch.device(args.device))
    demo.launch(server_name=HOST, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
