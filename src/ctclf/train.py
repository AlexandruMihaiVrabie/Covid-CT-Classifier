"""Training loop with per-epoch checkpoints and resume."""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from ctclf import config
from ctclf.data import build_datasets
from ctclf.model import build_model, save_weights

BEST_NAME = "best.pth"
LAST_NAME = "last.pt"
HISTORY_NAME = "train_history.json"


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    """DataLoader with settings suited to a small Mac (no pinned memory)."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """One pass over the loader; returns mean loss and accuracy."""
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss: {loss.item()}")
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        seen += labels.size(0)
    return {"loss": total_loss / seen, "acc": correct / seen}


@torch.no_grad()
def evaluate_loader(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> dict[str, float]:
    """Mean loss, accuracy and macro-F1 over a loader."""
    model.eval()
    total_loss, preds, targets = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * labels.size(0)
        preds.extend(logits.argmax(1).cpu().tolist())
        targets.extend(labels.cpu().tolist())
    correct = sum(p == t for p, t in zip(preds, targets, strict=True))
    return {
        "loss": total_loss / len(targets),
        "acc": correct / len(targets),
        "macro_f1": f1_score(targets, preds, average="macro", zero_division=0),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_f1: float,
    history: list[dict],
) -> None:
    """Save full resume state (weights, optimizer, epoch, best score, history)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_f1": best_f1,
            "history": history,
        },
        tmp,
    )
    tmp.replace(path)  # atomic, so an interrupted save never corrupts last.pt


def load_checkpoint(
    path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device
) -> tuple[int, float, list[dict]]:
    """Restore resume state in place; returns (last completed epoch, best_f1, history)."""
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return state["epoch"], state["best_f1"], state["history"]


def fit(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    epochs: int,
    checkpoint_dir: Path,
    results_dir: Path,
    device: torch.device,
    resume: bool = False,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    lr: float = config.LEARNING_RATE,
    weight_decay: float = config.WEIGHT_DECAY,
) -> list[dict]:
    """Train for `epochs` total epochs, checkpointing every epoch; returns the history."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    last_path = Path(checkpoint_dir) / LAST_NAME
    start_epoch, best_f1, history = 0, -math.inf, []
    if resume and last_path.exists():
        start_epoch, best_f1, history = load_checkpoint(last_path, model, optimizer, device)
        print(f"Resumed from {last_path} after epoch {start_epoch}")

    train_loader = make_loader(train_ds, batch_size, True, num_workers)
    val_loader = make_loader(val_ds, batch_size, False, num_workers)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch + 1, epochs + 1):
        t0 = time.perf_counter()
        train_m = train_one_epoch(model, train_loader, optimizer, criterion, device)
        t1 = time.perf_counter()
        val_m = evaluate_loader(model, val_loader, criterion, device)
        t2 = time.perf_counter()

        row = {
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "train_acc": train_m["acc"],
            "val_loss": val_m["loss"],
            "val_acc": val_m["acc"],
            "val_macro_f1": val_m["macro_f1"],
            "train_seconds": t1 - t0,
            "val_seconds": t2 - t1,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
        }
        is_best = val_m["macro_f1"] > best_f1
        if is_best:
            best_f1 = val_m["macro_f1"]
            save_weights(model, Path(checkpoint_dir) / BEST_NAME)
        row["is_best"] = is_best
        history.append(row)
        save_checkpoint(last_path, model, optimizer, epoch, best_f1, history)
        (results_dir / HISTORY_NAME).write_text(json.dumps(history, indent=2))
        print(
            f"epoch {epoch}/{epochs} | train loss {row['train_loss']:.4f} "
            f"acc {row['train_acc']:.4f} | val loss {row['val_loss']:.4f} "
            f"acc {row['val_acc']:.4f} f1 {row['val_macro_f1']:.4f}"
            f"{' *best*' if is_best else ''} | {t1 - t0:.0f}s + {t2 - t1:.0f}s",
            flush=True,
        )
    return history


def _subset(ds: Dataset, limit: int | None, seed: int) -> Dataset:
    """Random fixed-seed subset of at most `limit` items (None = full dataset)."""
    if limit is None or limit >= len(ds):
        return ds
    return Subset(ds, random.Random(seed).sample(range(len(ds)), limit))


def main() -> None:
    """Train MobileNetV3-Large on the CT slices (checkpoint every epoch, resumable)."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--limit-train", type=int, help="random subset size (smoke tests)")
    parser.add_argument("--limit-val", type=int, help="random subset size (smoke tests)")
    parser.add_argument("--checkpoint-dir", type=Path, default=config.CHECKPOINTS_DIR)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--device", help="override auto-detection, e.g. cpu")
    args = parser.parse_args()

    torch.manual_seed(config.SEED)
    device = torch.device(args.device) if args.device else config.get_device()
    datasets = build_datasets()
    train_ds = _subset(datasets["train"], args.limit_train, config.SEED)
    val_ds = _subset(datasets["val"], args.limit_val, config.SEED)
    print(f"device={device} train={len(train_ds)} val={len(val_ds)} epochs={args.epochs}")

    fit(
        build_model(pretrained=True),
        train_ds,
        val_ds,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        device=device,
        resume=args.resume,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
