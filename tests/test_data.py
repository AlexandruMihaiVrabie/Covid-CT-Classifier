"""Tests for the dataset index, split, transforms and verification."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from ctclf import config
from ctclf.data import (
    REFERENCE_COUNTS,
    REFERENCE_SPLIT_SIZES,
    CTSliceDataset,
    Sample,
    build_index,
    duplicates_across_splits,
    get_transforms,
    split_index,
    split_sizes,
    verify_dataset,
)

FAKE_COUNTS = {"nCT": 40, "pCT": 17, "NiCT": 23}


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    """Tiny dataset with unique images per file."""
    for c, (cls, n) in enumerate(FAKE_COUNTS.items()):
        (tmp_path / cls).mkdir()
        for i in range(1, n + 1):
            # Unique colour per file so there are no byte-level duplicates.
            colour = (i * 5, c * 80, 100)
            Image.new("RGB", (300, 280), colour).save(tmp_path / cls / f"{cls}{i}.jpg")
    return tmp_path


def _paths(samples) -> set[Path]:
    return {s.path for s in samples}


def test_build_index_counts_and_labels(fake_root: Path) -> None:
    samples = build_index(fake_root)
    assert len(samples) == sum(FAKE_COUNTS.values())
    for label, cls in enumerate(config.CLASSES):
        assert sum(s.label == label for s in samples) == FAKE_COUNTS[cls]
        assert all(s.path.parent.name == cls for s in samples if s.label == label)


def test_build_index_missing_class(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_index(tmp_path)


def test_split_sizes_reproduce_original_repo() -> None:
    totals = [
        sum(x) for x in zip(*(split_sizes(n) for n in REFERENCE_COUNTS.values()), strict=True)
    ]
    assert totals == [REFERENCE_SPLIT_SIZES[s] for s in ("train", "val", "test")]


def test_split_has_no_overlap_and_covers_everything(fake_root: Path) -> None:
    samples = build_index(fake_root)
    splits = split_index(samples)
    train, val, test = (_paths(splits[s]) for s in ("train", "val", "test"))
    assert not train & val and not train & test and not val & test
    assert train | val | test == _paths(samples)


def test_split_is_deterministic_and_order_independent(fake_root: Path) -> None:
    samples = build_index(fake_root)
    a = split_index(samples)
    b = split_index(list(reversed(samples)))
    assert {k: [s.path for s in v] for k, v in a.items()} == {
        k: [s.path for s in v] for k, v in b.items()
    }


def test_split_changes_with_seed(fake_root: Path) -> None:
    samples = build_index(fake_root)
    assert split_index(samples, seed=1)["test"] != split_index(samples, seed=2)["test"]


def test_split_is_stratified(fake_root: Path) -> None:
    splits = split_index(build_index(fake_root))
    for label, cls in enumerate(config.CLASSES):
        got = tuple(sum(s.label == label for s in splits[k]) for k in ("train", "val", "test"))
        assert got == split_sizes(FAKE_COUNTS[cls])


@pytest.mark.parametrize("train", [True, False])
def test_dataset_item_shape(fake_root: Path, train: bool) -> None:
    ds = CTSliceDataset(build_index(fake_root), get_transforms(train=train))
    image, label = ds[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    assert label == 0


def test_grayscale_image_becomes_rgb(tmp_path: Path) -> None:
    path = tmp_path / "gray.png"
    Image.new("L", (256, 256), 128).save(path)
    image, _ = CTSliceDataset([Sample(path, 0)], get_transforms(train=False))[0]
    assert image.shape[0] == 3


def test_verify_detects_unreadable_and_duplicates(fake_root: Path) -> None:
    (fake_root / "nCT" / "nCT999.jpg").write_bytes(b"not an image")
    dup_src = fake_root / "pCT" / "pCT1.jpg"
    (fake_root / "NiCT" / "NiCT999.jpg").write_bytes(dup_src.read_bytes())

    report = verify_dataset(build_index(fake_root))
    assert len(report["unreadable"]) == 1
    assert report["duplicate_groups"] == 1
    assert report["cross_class_duplicate_groups"] == 1


def test_duplicates_across_splits_zero_for_unique_images(fake_root: Path) -> None:
    leaks = duplicates_across_splits(split_index(build_index(fake_root)))
    assert set(leaks.values()) == {0}


@pytest.mark.skipif(not config.DATA_ROOT.is_dir(), reason="real dataset not present")
def test_real_dataset_split() -> None:
    samples = build_index()
    a, b = split_index(samples), split_index(build_index())
    assert {k: len(v) for k, v in a.items()} == REFERENCE_SPLIT_SIZES
    assert all([s.path for s in a[k]] == [s.path for s in b[k]] for k in a)
    train, val, test = (_paths(a[s]) for s in ("train", "val", "test"))
    assert not train & val and not train & test and not val & test
