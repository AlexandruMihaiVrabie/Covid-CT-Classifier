"""Evaluation metrics, plots and majority-class baseline."""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: plots are only written to disk
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from ctclf import config  # noqa: E402
from ctclf.data import build_datasets, get_transforms  # noqa: E402
from ctclf.model import load_model  # noqa: E402
from ctclf.train import BEST_NAME, HISTORY_NAME, make_loader  # noqa: E402

METRICS_NAME = "metrics.json"
CURVES_NAME = "training_curves.png"
CONFUSION_NAME = "confusion_matrix_test.png"
# Reported by the original repo (25 epochs); shown for reference only, not a target.
ORIGINAL_TEST_ACCURACY = 0.9976


def classification_metrics(
    targets: list[int], preds: list[int], classes: tuple[str, ...] = config.CLASSES
) -> dict:
    """Accuracy, macro-F1, per-class precision/recall/F1 and confusion matrix."""
    labels = list(range(len(classes)))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, preds, labels=labels, zero_division=0
    )
    return {
        "n": len(targets),
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(
            f1_score(targets, preds, labels=labels, average="macro", zero_division=0)
        ),
        "per_class": {
            cls: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, cls in enumerate(classes)
        },
        "confusion_matrix": confusion_matrix(targets, preds, labels=labels).tolist(),
    }


def majority_baseline(
    train_targets: list[int], test_targets: list[int], classes: tuple[str, ...] = config.CLASSES
) -> dict:
    """Always predict the most frequent training class; score it on the test labels."""
    majority = Counter(train_targets).most_common(1)[0][0]
    metrics = classification_metrics(test_targets, [majority] * len(test_targets), classes)
    return {"predicted_class": classes[majority], **metrics}


def best_history_row(history: list[dict]) -> dict:
    """The last epoch flagged as best (the one whose weights are in best.pth)."""
    best = [row for row in history if row.get("is_best")]
    if not best:
        raise ValueError("No epoch is flagged as best in the training history")
    return best[-1]


def consistency_check(recomputed: dict, history_row: dict, tol: float = 2e-3) -> dict:
    """Compare val metrics recomputed from best.pth with those logged during training."""
    pairs = {
        "loss": (recomputed["loss"], history_row["val_loss"]),
        "accuracy": (recomputed["accuracy"], history_row["val_acc"]),
        "macro_f1": (recomputed["macro_f1"], history_row["val_macro_f1"]),
    }
    diffs = {k: abs(a - b) for k, (a, b) in pairs.items()}
    return {
        "epoch": history_row["epoch"],
        "tolerance": tol,
        "recomputed": {k: a for k, (a, _) in pairs.items()},
        "training_log": {k: b for k, (_, b) in pairs.items()},
        "abs_diff": diffs,
        "passed": all(d <= tol for d in diffs.values()),
    }


@torch.no_grad()
def predict_loader(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, list[int], list[int]]:
    """Mean cross-entropy loss, predictions and targets over a loader."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, preds, targets = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item()
        preds.extend(logits.argmax(1).cpu().tolist())
        targets.extend(labels.cpu().tolist())
    return total_loss / len(targets), preds, targets


def plot_training_curves(history: list[dict], path: Path) -> None:
    """Loss, accuracy and val macro-F1 per epoch."""
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [r["train_loss"] for r in history], marker="o", label="train")
    axes[0].plot(epochs, [r["val_loss"] for r in history], marker="o", label="val")
    axes[0].set_title("Cross-entropy loss")
    axes[1].plot(epochs, [r["train_acc"] for r in history], marker="o", label="train")
    axes[1].plot(epochs, [r["val_acc"] for r in history], marker="o", label="val")
    axes[1].set_title("Accuracy")
    axes[2].plot(epochs, [r["val_macro_f1"] for r in history], marker="o", label="val")
    axes[2].set_title("Validation macro-F1")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(
    matrix: list[list[int]], path: Path, classes: tuple[str, ...] = config.CLASSES
) -> None:
    """Annotated confusion matrix (rows = true class, columns = predicted)."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    peak = max(max(row) for row in matrix) or 1
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            color = "white" if value > peak / 2 else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax)
    ax.set_title("Test confusion matrix")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def evaluate_split(model: nn.Module, ds: Dataset, device: torch.device, num_workers: int) -> dict:
    """Loss plus classification metrics for one dataset."""
    loader = make_loader(ds, config.BATCH_SIZE, False, num_workers)
    loss, preds, targets = predict_loader(model, loader, device)
    return {"loss": loss, **classification_metrics(targets, preds)}


def main() -> None:
    """Evaluate best.pth on train/val/test and write metrics and plots."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--checkpoint", type=Path, default=config.CHECKPOINTS_DIR / BEST_NAME)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--device", help="override auto-detection, e.g. cpu")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else config.get_device()
    model = load_model(args.checkpoint, device)
    datasets = build_datasets()
    # Train split is scored with eval transforms, so the numbers do not depend on augmentation.
    datasets["train"].transform = get_transforms(train=False)

    results: dict = {"checkpoint": str(args.checkpoint), "device": str(device), "splits": {}}
    for name, ds in datasets.items():
        m = evaluate_split(model, ds, device, args.num_workers)
        results["splits"][name] = m
        print(
            f"{name:5s} n={m['n']:5d} loss {m['loss']:.4f} acc {m['accuracy']:.4f} "
            f"macro-F1 {m['macro_f1']:.4f}",
            flush=True,
        )

    train_targets = [s.label for s in datasets["train"].samples]
    test_targets = [s.label for s in datasets["test"].samples]
    results["majority_baseline_test"] = majority_baseline(train_targets, test_targets)
    results["original_repo_test_accuracy_reference"] = ORIGINAL_TEST_ACCURACY

    results_dir = Path(args.results_dir)
    history = json.loads((results_dir / HISTORY_NAME).read_text())
    check = consistency_check(results["splits"]["val"], best_history_row(history))
    results["consistency_check"] = check

    (results_dir / METRICS_NAME).write_text(json.dumps(results, indent=2))
    plot_training_curves(history, results_dir / CURVES_NAME)
    plot_confusion_matrix(
        results["splits"]["test"]["confusion_matrix"], results_dir / CONFUSION_NAME
    )

    test = results["splits"]["test"]
    base = results["majority_baseline_test"]
    print("\nTest per class:")
    for cls, pc in test["per_class"].items():
        print(
            f"  {cls:5s} P {pc['precision']:.4f} R {pc['recall']:.4f} F1 {pc['f1']:.4f} "
            f"(n={pc['support']})"
        )
    print(
        f"Test confusion matrix (rows true, cols pred {config.CLASSES}): {test['confusion_matrix']}"
    )
    print(
        f"Majority baseline (always {base['predicted_class']}): acc {base['accuracy']:.4f} "
        f"macro-F1 {base['macro_f1']:.4f}"
    )
    print(f"Original repo test accuracy (reference only): {ORIGINAL_TEST_ACCURACY:.4f}")
    print(
        f"Consistency check vs training log (epoch {check['epoch']}): "
        f"{'PASS' if check['passed'] else 'FAIL'} abs_diff={check['abs_diff']}"
    )
    print(f"Wrote {results_dir / METRICS_NAME}, {CURVES_NAME}, {CONFUSION_NAME}")


if __name__ == "__main__":
    main()
