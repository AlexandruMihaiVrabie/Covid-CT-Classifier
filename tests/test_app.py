"""Tests for the Gradio app logic (random weights, synthetic images, no launch, no network)."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import pytest
import torch
from PIL import Image

from ctclf import config
from ctclf.model import build_model, save_weights

APP_PATH = config.PROJECT_ROOT / "app" / "app.py"


def load_app_module():
    spec = importlib.util.spec_from_file_location("ctclf_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def app_module():
    return load_app_module()


@pytest.fixture(scope="module")
def model() -> torch.nn.Module:
    torch.manual_seed(0)
    return build_model(pretrained=False).eval()


def test_predict_outputs(app_module, model: torch.nn.Module) -> None:
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, (300, 260), dtype=np.uint8), mode="L")
    probs, overlay, text = app_module.make_predict_fn(model)(image)
    assert list(probs) == list(config.CLASSES)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-5)
    assert overlay.shape == (config.IMAGE_SIZE, config.IMAGE_SIZE, 3)
    assert overlay.dtype == np.uint8
    assert text.startswith(f"**Prediction: {max(probs, key=probs.get)}**")


def test_predict_without_image_raises(app_module, model: torch.nn.Module) -> None:
    with pytest.raises(gr.Error):
        app_module.make_predict_fn(model)(None)


def test_build_app_and_examples(app_module, tmp_path: Path) -> None:
    checkpoint = tmp_path / "w.pth"
    save_weights(build_model(pretrained=False), checkpoint)
    assert isinstance(app_module.build_app(checkpoint), gr.Blocks)
    for name, cls in zip(app_module.EXAMPLES, config.CLASSES, strict=True):
        assert (app_module.EXAMPLES_DIR / name).exists()
        assert name.startswith(cls)


def test_disclaimer_and_localhost(app_module) -> None:
    assert "not a medical device" in app_module.DISCLAIMER
    assert "even for images that are not chest-CT" in app_module.ALWAYS_A_CLASS_NOTE
    assert app_module.HOST == "127.0.0.1"


def test_smoke_check_inputs(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("smoke", APP_PATH.parent / "smoke_check.py")
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    colour, gray = smoke.non_ct_images(tmp_path)
    assert Image.open(colour).mode == "RGB" and Image.open(gray).mode == "L"
    assert all((smoke.EXAMPLES_DIR / n).exists() for n in smoke.EXAMPLES)


def test_app_does_not_load_api_or_env_modules() -> None:
    code = (
        "import importlib.util, sys;"
        f"spec = importlib.util.spec_from_file_location('a', r'{APP_PATH}');"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m);"
        "bad = [n for n in ('dotenv', 'openai', 'ctclf.nvidia_client') if n in sys.modules];"
        "print(bad)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]"
