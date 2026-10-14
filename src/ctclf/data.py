"""Dataset index, deterministic split, datasets and transforms."""

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms as T

from ctclf import config

SPLITS: tuple[str, ...] = ("train", "val", "test")
# Reference numbers from the original repo's README.
REFERENCE_COUNTS = {"nCT": 9979, "pCT": 4001, "NiCT": 5705}
REFERENCE_SPLIT_SIZES = {"train": 13778, "val": 2951, "test": 2956}


@dataclass(frozen=True)
class Sample:
    """One CT slice: file path and integer label."""

    path: Path
    label: int


def _natural_key(path: Path) -> tuple[str, int]:
    """Sort key so that nCT2 comes before nCT10."""
    match = re.match(r"^(\D*)(\d+)", path.stem)
    if match:
        return match.group(1), int(match.group(2))
    return path.stem, -1


def build_index(
    root: Path = config.DATA_ROOT, classes: tuple[str, ...] = config.CLASSES
) -> list[Sample]:
    """List every image under root/<class>/, sorted, labelled by class order."""
    samples: list[Sample] = []
    for label, cls in enumerate(classes):
        class_dir = Path(root) / cls
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        files = [
            p
            for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
        ]
        samples.extend(Sample(p, label) for p in sorted(files, key=_natural_key))
    return samples


def split_sizes(
    n: int,
    train_fraction: float = config.TRAIN_FRACTION,
    val_fraction: float = config.VAL_FRACTION,
) -> tuple[int, int, int]:
    """Per-class (train, val, test) counts: truncate train and val, test gets the rest."""
    n_train = int(n * train_fraction)
    n_val = int(n * val_fraction)
    return n_train, n_val, n - n_train - n_val


def split_index(
    samples: list[Sample],
    seed: int = config.SEED,
    train_fraction: float = config.TRAIN_FRACTION,
    val_fraction: float = config.VAL_FRACTION,
) -> dict[str, list[Sample]]:
    """Deterministic per-class (stratified) split into train/val/test."""
    by_label: dict[int, list[Sample]] = defaultdict(list)
    for s in samples:
        by_label[s.label].append(s)

    splits: dict[str, list[Sample]] = {name: [] for name in SPLITS}
    for label in sorted(by_label):
        # Sort first so the result does not depend on the input order.
        group = sorted(by_label[label], key=lambda s: _natural_key(s.path))
        rng = random.Random(seed + label)
        rng.shuffle(group)
        n_train, n_val, _ = split_sizes(len(group), train_fraction, val_fraction)
        splits["train"].extend(group[:n_train])
        splits["val"].extend(group[n_train : n_train + n_val])
        splits["test"].extend(group[n_train + n_val :])
    return splits


def get_transforms(train: bool) -> T.Compose:
    """Train: random resized crop + flip; eval: resize 256 + center crop 224."""
    normalize = T.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)
    if train:
        return T.Compose(
            [
                T.RandomResizedCrop(config.IMAGE_SIZE),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize,
            ]
        )
    return T.Compose(
        [
            T.Resize(config.RESIZE_SIZE),
            T.CenterCrop(config.IMAGE_SIZE),
            T.ToTensor(),
            normalize,
        ]
    )


def load_image(path: Path) -> Image.Image:
    """Open an image as 3-channel RGB."""
    with Image.open(path) as img:
        return img.convert("RGB")


class CTSliceDataset(Dataset):
    """Torch dataset over a list of samples."""

    def __init__(self, samples: list[Sample], transform: T.Compose | None = None) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        sample = self.samples[idx]
        image = load_image(sample.path)
        if self.transform is not None:
            image = self.transform(image)
        return image, sample.label


def build_datasets(
    root: Path = config.DATA_ROOT, seed: int = config.SEED
) -> dict[str, CTSliceDataset]:
    """Index, split and wrap into train/val/test datasets with their transforms."""
    splits = split_index(build_index(root), seed=seed)
    return {
        name: CTSliceDataset(items, get_transforms(train=name == "train"))
        for name, items in splits.items()
    }


def verify_dataset(samples: list[Sample], classes: tuple[str, ...] = config.CLASSES) -> dict:
    """Check readability, image sizes/modes and exact (byte-level) duplicates."""
    unreadable: list[str] = []
    sizes: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    by_hash: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        data = s.path.read_bytes()
        by_hash[hashlib.sha256(data).hexdigest()].append(s)
        try:
            with Image.open(s.path) as img:
                img.load()
                sizes[f"{img.width}x{img.height}"] += 1
                modes[img.mode] += 1
        except Exception as exc:  # any decode failure counts as unreadable
            unreadable.append(f"{s.path.name}: {exc}")

    dup_groups = [g for g in by_hash.values() if len(g) > 1]
    cross_class = [g for g in dup_groups if len({s.label for s in g}) > 1]
    return {
        "counts": {cls: sum(s.label == i for s in samples) for i, cls in enumerate(classes)},
        "total": len(samples),
        "unreadable": unreadable,
        "image_modes": dict(modes),
        "distinct_sizes": len(sizes),
        "most_common_sizes": dict(sizes.most_common(5)),
        "duplicate_groups": len(dup_groups),
        "duplicate_extra_files": sum(len(g) - 1 for g in dup_groups),
        "cross_class_duplicate_groups": len(cross_class),
        "duplicate_examples": [[s.path.name for s in g] for g in dup_groups[:10]],
    }


def duplicates_across_splits(splits: dict[str, list[Sample]]) -> dict[str, int]:
    """Count byte-identical images shared between each pair of splits (leakage check)."""
    hashes = {
        name: {hashlib.sha256(s.path.read_bytes()).hexdigest() for s in items}
        for name, items in splits.items()
    }
    return {
        f"{a}-{b}": len(hashes[a] & hashes[b])
        for i, a in enumerate(SPLITS)
        for b in SPLITS[i + 1 :]
    }


def filename_patterns(samples: list[Sample]) -> dict[str, int]:
    """Group filenames by pattern (digits replaced by #) to look for patient identifiers."""
    return dict(Counter(re.sub(r"\d+", "#", s.path.name) for s in samples))


def main() -> None:
    """Verify the dataset, print the report and split sizes, save them to outputs/results."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--root", type=Path, default=config.DATA_ROOT)
    args = parser.parse_args()

    samples = build_index(args.root)
    report = verify_dataset(samples)
    report["reference_counts"] = REFERENCE_COUNTS
    report["filename_patterns"] = filename_patterns(samples)

    splits = split_index(samples)
    report["split_sizes"] = {name: len(items) for name, items in splits.items()}
    report["split_sizes_per_class"] = {
        name: {cls: sum(s.label == i for s in items) for i, cls in enumerate(config.CLASSES)}
        for name, items in splits.items()
    }
    report["reference_split_sizes"] = REFERENCE_SPLIT_SIZES
    report["duplicates_across_splits"] = duplicates_across_splits(splits)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / "data_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved to {out.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
