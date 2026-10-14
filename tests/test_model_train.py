"""Tests for the model definition and the training loop (CPU, no pretrained download)."""

import math
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from ctclf.model import build_model, load_model, save_weights
from ctclf.train import HISTORY_NAME, LAST_NAME, evaluate_loader, fit, make_loader

DEVICE = torch.device("cpu")


@pytest.fixture
def tiny_ds() -> TensorDataset:
    """Small random dataset (64x64 keeps the CPU tests fast)."""
    g = torch.Generator().manual_seed(0)
    images = torch.randn(12, 3, 64, 64, generator=g)
    labels = torch.tensor([0, 1, 2] * 4)
    return TensorDataset(images, labels)


def test_output_shape() -> None:
    model = build_model(pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(4, 3, 224, 224))
    assert out.shape == (4, 3)


def test_classifier_head() -> None:
    head = build_model(pretrained=False).classifier
    assert [type(m) for m in head] == [nn.Linear, nn.ReLU, nn.Dropout, nn.Linear]
    assert (head[0].in_features, head[0].out_features) == (960, 1280)
    assert (head[3].in_features, head[3].out_features) == (1280, 3)


def test_save_load_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = build_model(pretrained=False).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        before = model(x)
    save_weights(model, tmp_path / "w.pth")
    with torch.no_grad():
        after = load_model(tmp_path / "w.pth")(x)
    assert torch.equal(before, after)


def test_can_overfit_tiny_batch(tiny_ds: TensorDataset) -> None:
    torch.manual_seed(0)
    model = build_model(pretrained=False)
    model.classifier[2].p = 0.0  # no dropout, so memorising is deterministic
    images, labels = tiny_ds.tensors
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model.train()
    losses = []
    for _ in range(60):
        opt.zero_grad()
        loss = criterion(model(images), labels)
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.05 < losses[0]


def test_fit_checkpoints_and_resume(tiny_ds: TensorDataset, tmp_path: Path) -> None:
    torch.manual_seed(0)
    kwargs = dict(
        checkpoint_dir=tmp_path / "ckpt",
        results_dir=tmp_path / "res",
        device=DEVICE,
        batch_size=4,
        num_workers=0,
    )
    history = fit(build_model(pretrained=False), tiny_ds, tiny_ds, epochs=1, **kwargs)
    assert len(history) == 1 and history[0]["is_best"]
    assert (tmp_path / "ckpt" / LAST_NAME).exists()
    assert (tmp_path / "ckpt" / "best.pth").exists()
    assert (tmp_path / "res" / HISTORY_NAME).exists()

    history = fit(build_model(pretrained=False), tiny_ds, tiny_ds, epochs=2, resume=True, **kwargs)
    assert [row["epoch"] for row in history] == [1, 2]
    assert all(math.isfinite(row["train_loss"]) for row in history)


def test_evaluate_loader_metrics(tiny_ds: TensorDataset) -> None:
    model = build_model(pretrained=False)
    metrics = evaluate_loader(
        model, make_loader(tiny_ds, 4, False, 0), nn.CrossEntropyLoss(), DEVICE
    )
    assert 0.0 <= metrics["acc"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert math.isfinite(metrics["loss"])
