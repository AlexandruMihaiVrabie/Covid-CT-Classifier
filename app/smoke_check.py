"""Smoke check of a running app through gradio_client: three examples plus two non-CT images."""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
from gradio_client import Client, handle_file
from PIL import Image

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "assets" / "examples"
EXAMPLES = ("nCT7356.jpg", "pCT2369.jpg", "NiCT219.jpg")


def non_ct_images(out_dir: Path) -> list[Path]:
    """A colourful synthetic picture (RGB) and a grayscale gradient: clearly not CT slices."""
    yy, xx = np.mgrid[0:400, 0:600]
    rgb = np.stack([xx * 255 // 600, yy * 255 // 400, (xx + yy) % 256], axis=-1)
    colour, gray = out_dir / "non_ct_colour.png", out_dir / "non_ct_gray.png"
    Image.fromarray(rgb.astype(np.uint8)).save(colour)
    Image.fromarray((xx[:300, :300] % 256).astype(np.uint8), mode="L").save(gray)
    return [colour, gray]


def main() -> None:
    """Call /predict on each input and check the label, probabilities and overlay shape."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860/")
    args = parser.parse_args()

    client = Client(args.url, verbose=False)
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        inputs = [EXAMPLES_DIR / n for n in EXAMPLES] + non_ct_images(Path(tmp))
        for path in inputs:
            label, cam_path, text = client.predict(handle_file(str(path)), api_name="/predict")
            confs = {c["label"]: c["confidence"] for c in label["confidences"]}
            cam = np.asarray(Image.open(cam_path))
            good = (
                len(confs) == 3
                and abs(sum(confs.values()) - 1) < 1e-4
                and cam.shape == (224, 224, 3)
            )
            ok &= good
            probs = ", ".join(f"{k} {v:.4f}" for k, v in confs.items())
            print(f"{path.name:18s} -> {label['label']:4s} [{probs}] overlay {cam.shape} "
                  f"{'OK' if good else 'BAD'}")  # fmt: skip
            print(f"{'':22s}{text}")
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
