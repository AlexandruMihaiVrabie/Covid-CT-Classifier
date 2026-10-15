"""Tests for evaluation metrics, baseline, consistency check and plots (no dataset)."""

from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from ctclf.evaluate import (
    best_history_row,
    classification_metrics,
    consistency_check,
    evaluate_split,
    majority_baseline,
    plot_confusion_matrix,
    plot_training_curves,
)
from ctclf.model import build_model


def test_classification_metrics_known_values() -> None:
    targets = [0, 0, 1, 1, 2, 2]
    preds = [0, 1, 1, 1, 2, 0]
    m = classification_metrics(targets, preds)
    assert m["n"] == 6
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert m["per_class"]["pCT"]["precision"] == pytest.approx(2 / 3)
    assert m["per_class"]["pCT"]["recall"] == pytest.approx(1.0)
    # F1 per class: nCT 0.5, pCT 0.8, NiCT 2/3
    assert m["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)


def test_majority_baseline_uses_train_majority() -> None:
    base = majority_baseline(train_targets=[0, 0, 0, 1, 2], test_targets=[0, 1, 2, 0])
    assert base["predicted_class"] == "nCT"
    assert base["accuracy"] == pytest.approx(0.5)
    assert base["per_class"]["pCT"]["recall"] == 0.0


def _history() -> list[dict]:
    rows = []
    for epoch, f1 in [(1, 0.8), (2, 0.9), (3, 0.85)]:
        rows.append(
            {
                "epoch": epoch,
                "train_loss": 1.0 / epoch,
                "train_acc": f1,
                "val_loss": 0.5 / epoch,
                "val_acc": f1,
                "val_macro_f1": f1,
                "is_best": epoch in (1, 2),
            }
        )
    return rows


def test_best_history_row_picks_last_best() -> None:
    assert best_history_row(_history())["epoch"] == 2
    with pytest.raises(ValueError):
        best_history_row([{"epoch": 1, "is_best": False}])


def test_consistency_check_pass_and_fail() -> None:
    row = best_history_row(_history())
    ok = consistency_check({"loss": 0.2501, "accuracy": 0.9, "macro_f1": 0.9}, row)
    assert ok["passed"]
    bad = consistency_check({"loss": 0.25, "accuracy": 0.95, "macro_f1": 0.9}, row)
    assert not bad["passed"]


def test_plots_write_files(tmp_path: Path) -> None:
    plot_training_curves(_history(), tmp_path / "curves.png")
    plot_confusion_matrix([[5, 1, 0], [0, 4, 1], [0, 0, 6]], tmp_path / "cm.png")
    assert (tmp_path / "curves.png").stat().st_size > 0
    assert (tmp_path / "cm.png").stat().st_size > 0


def test_evaluate_split_on_tiny_dataset() -> None:
    torch.manual_seed(0)
    ds = TensorDataset(torch.randn(6, 3, 64, 64), torch.tensor([0, 1, 2] * 2))
    m = evaluate_split(build_model(pretrained=False), ds, torch.device("cpu"), num_workers=0)
    assert m["n"] == 6
    assert m["loss"] > 0
    assert sum(map(sum, m["confusion_matrix"])) == 6
