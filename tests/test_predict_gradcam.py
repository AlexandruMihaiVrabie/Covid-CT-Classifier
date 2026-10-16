"""Tests for prediction and Grad-CAM (random weights, synthetic images, no dataset)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ctclf import config, gradcam, predict
from ctclf.model import build_model, save_weights


@pytest.fixture(scope="module")
def model() -> torch.nn.Module:
    torch.manual_seed(0)
    return build_model(pretrained=False).eval()


@pytest.fixture
def image() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (300, 340), dtype=np.uint8), mode="L")


def test_preprocess_shape(image: Image.Image) -> None:
    assert predict.preprocess(image).shape == (1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)


def test_probabilities_sum_to_one(model: torch.nn.Module, image: Image.Image) -> None:
    result = predict.predict_image(model, image)
    probs = result["probabilities"]
    assert list(probs) == list(config.CLASSES)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-5)
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert result["label"] == max(probs, key=probs.get)
    assert result["description"] == config.CLASS_DESCRIPTIONS[result["label"]]


def test_heatmap_matches_input_size(model: torch.nn.Module, image: Image.Image) -> None:
    heatmap, cls = gradcam.compute_gradcam(model, image, target_class=1)
    assert cls == 1
    assert heatmap.shape == (config.IMAGE_SIZE, config.IMAGE_SIZE)
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0
    assert heatmap.shape == predict.preprocess(image).shape[-2:]


def test_overlay_is_rgb_uint8(model: torch.nn.Module, image: Image.Image) -> None:
    prediction, blended = gradcam.explain_image(model, image)
    assert blended.shape == (config.IMAGE_SIZE, config.IMAGE_SIZE, 3)
    assert blended.dtype == np.uint8
    assert prediction["label"] in config.CLASSES


def test_target_layer_is_last_conv_block(model: torch.nn.Module) -> None:
    layer = gradcam.get_target_layer(model)
    assert layer is model.features[-1]
    assert layer[0].out_channels == 960


def test_save_examples_writes_files(
    model: torch.nn.Module, image: Image.Image, tmp_path: Path
) -> None:
    paths = []
    for cls in config.CLASSES:
        (tmp_path / cls).mkdir()
        paths.append(tmp_path / cls / f"{cls}1.jpg")
        image.save(paths[-1])
    written = gradcam.save_examples(model, paths, tmp_path / "out")
    assert [p.name for p in written] == [
        "gradcam_nCT.png",
        "gradcam_pCT.png",
        "gradcam_NiCT.png",
        gradcam.EXAMPLES_NAME,
    ]
    assert all(p.stat().st_size > 0 for p in written)


def test_predict_cli_json(
    model: torch.nn.Module,
    image: Image.Image,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ckpt, img_path = tmp_path / "w.pth", tmp_path / "slice.png"
    save_weights(model, ckpt)
    image.save(img_path)
    argv = ["predict", str(img_path), "--checkpoint", str(ckpt), "--device", "cpu", "--json"]
    monkeypatch.setattr(sys, "argv", argv)
    predict.main()
    out = capsys.readouterr().out
    assert '"probabilities"' in out
    # Same weights in memory and from disk give the same answer.
    assert predict.predict_path(img_path, ckpt)["probabilities"] == pytest.approx(
        predict.predict_image(model, image)["probabilities"]
    )
