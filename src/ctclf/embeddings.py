"""Frozen-encoder image embeddings: timing test and resumable cached extraction."""

import argparse
import csv
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ctclf import config
from ctclf.data import SPLITS, CTSliceDataset, Sample, build_index, split_index

EMBEDDINGS_DIR = config.RESULTS_DIR / "embeddings"
DEFAULT_ENCODER = "biomedclip"
# Encoder name -> open_clip model id. Add entries here to swap in another encoder.
ENCODERS: dict[str, str] = {
    "biomedclip": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
}
BATCH_SIZE = 32
CHUNK_SIZE = 512  # images per cached part file; the unit of resumption
INDEX_FIELDS = ("filename", "split", "label", "label_index")


@dataclass
class Encoder:
    """A frozen image-text encoder with its own preprocessing and tokenizer."""

    name: str
    model: nn.Module
    preprocess: Callable
    tokenizer: Callable | None = None


def load_encoder(name: str = DEFAULT_ENCODER, device: torch.device | str = "cpu") -> Encoder:
    """Load a registered open_clip encoder (downloads weights on first use)."""
    import open_clip  # heavy import, only needed when a real encoder is used

    model_id = ENCODERS[name]
    model, preprocess = open_clip.create_model_from_pretrained(model_id)
    tokenizer = open_clip.get_tokenizer(model_id)
    return Encoder(name, model.to(device).eval(), preprocess, tokenizer)


@torch.no_grad()
def encode_samples(
    encoder: Encoder,
    samples: Sequence[Sample],
    device: torch.device | str = "cpu",
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
) -> np.ndarray:
    """Raw (unnormalised) float32 image embeddings, one row per sample, in order."""
    loader = DataLoader(
        CTSliceDataset(list(samples), encoder.preprocess),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    out = [encoder.model.encode_image(images.to(device)).float().cpu() for images, _ in loader]
    return torch.cat(out).numpy().astype(np.float32)


@torch.no_grad()
def encode_texts(
    encoder: Encoder, prompts: Sequence[str], device: torch.device | str = "cpu"
) -> np.ndarray:
    """Raw float32 text embeddings, one row per prompt."""
    tokens = encoder.tokenizer(list(prompts)).to(device)
    return encoder.model.encode_text(tokens).float().cpu().numpy().astype(np.float32)


def _paths(out_dir: Path, encoder_name: str) -> dict[str, Path]:
    return {
        "index": out_dir / f"{encoder_name}_index.csv",
        "parts": out_dir / f"{encoder_name}_parts",
        "text": out_dir / f"{encoder_name}_text.npz",
        "summary": out_dir / f"{encoder_name}_extraction.json",
        "timing": out_dir / f"{encoder_name}_timing.json",
    }


def split_file(out_dir: Path, encoder_name: str, split: str) -> Path:
    """Path of the merged embedding matrix for one split."""
    return out_dir / f"{encoder_name}_{split}.npy"


def index_rows(splits: dict[str, list[Sample]]) -> list[dict]:
    """One index row per sample, in split order then sample order."""
    return [
        {
            "filename": s.path.name,
            "split": name,
            "label": config.CLASSES[s.label],
            "label_index": s.label,
        }
        for name in SPLITS
        for s in splits.get(name, [])
    ]


def write_index(path: Path, rows: list[dict]) -> None:
    """Write the index CSV (filename, split, label, label_index)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_index(path: Path) -> list[dict]:
    """Read the index CSV back, with label_index as int."""
    with Path(path).open(newline="") as f:
        return [{**r, "label_index": int(r["label_index"])} for r in csv.DictReader(f)]


def extract_split(
    encoder: Encoder,
    samples: list[Sample],
    split: str,
    out_dir: Path,
    device: torch.device | str = "cpu",
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
    chunk_size: int = CHUNK_SIZE,
    log: Callable[[str], None] = print,
) -> tuple[np.ndarray, int]:
    """Embed one split chunk by chunk, skipping cached chunks; return (matrix, n computed)."""
    final = split_file(out_dir, encoder.name, split)
    if final.exists():
        emb = np.load(final)
        if len(emb) == len(samples):
            return emb, 0
    parts_dir = _paths(out_dir, encoder.name)["parts"]
    parts_dir.mkdir(parents=True, exist_ok=True)
    computed = 0
    chunks = []
    n_chunks = (len(samples) + chunk_size - 1) // chunk_size
    for k in range(n_chunks):
        chunk = samples[k * chunk_size : (k + 1) * chunk_size]
        part = parts_dir / f"{split}_{k:05d}.npy"
        if part.exists() and len(np.load(part)) == len(chunk):
            chunks.append(np.load(part))
            continue
        start = time.perf_counter()
        emb = encode_samples(encoder, chunk, device, batch_size, num_workers)
        np.save(part, emb)
        chunks.append(emb)
        computed += len(chunk)
        log(
            f"  {split} chunk {k + 1}/{n_chunks}: {len(chunk)} images in "
            f"{time.perf_counter() - start:.1f}s"
        )
    matrix = np.concatenate(chunks) if chunks else np.zeros((0, 0), dtype=np.float32)
    np.save(final, matrix)
    for k in range(n_chunks):
        (parts_dir / f"{split}_{k:05d}.npy").unlink(missing_ok=True)
    return matrix, computed


def load_split_embeddings(
    split: str,
    encoder_name: str = DEFAULT_ENCODER,
    out_dir: Path = EMBEDDINGS_DIR,
    expected: list[Sample] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Cached embeddings and index rows for one split, checked for alignment."""
    rows = [r for r in read_index(_paths(out_dir, encoder_name)["index"]) if r["split"] == split]
    emb = np.load(split_file(out_dir, encoder_name, split))
    if len(emb) != len(rows):
        raise ValueError(f"{split}: {len(emb)} embeddings but {len(rows)} index rows")
    if expected is not None:
        names = [s.path.name for s in expected]
        if names != [r["filename"] for r in rows]:
            raise ValueError(f"{split}: cached index does not match the current data split")
    return emb, rows


def cached_text_embeddings(
    prompts: Sequence[str],
    encoder_name: str = DEFAULT_ENCODER,
    out_dir: Path = EMBEDDINGS_DIR,
    encoder_factory: Callable[[], Encoder] | None = None,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Text embeddings for prompts, cached by prompt string; the encoder loads only on a miss."""
    path = _paths(out_dir, encoder_name)["text"]
    cache: dict[str, np.ndarray] = {}
    if path.exists():
        with np.load(path) as data:
            cache = dict(zip(data["prompts"].tolist(), data["embeddings"], strict=True))
    missing = [p for p in dict.fromkeys(prompts) if p not in cache]
    if missing:
        encoder = (encoder_factory or (lambda: load_encoder(encoder_name, device)))()
        cache.update(zip(missing, encode_texts(encoder, missing, device), strict=True))
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(cache)
        np.savez(path, prompts=np.array(keys), embeddings=np.stack([cache[k] for k in keys]))
    return np.stack([cache[p] for p in prompts])


def timing_test(
    encoder: Encoder,
    samples: list[Sample],
    n: int,
    device: torch.device | str,
    batch_size: int,
    num_workers: int,
    totals: dict[str, int],
) -> dict:
    """Time n images (after one warm-up batch) and extrapolate to the given totals."""
    step = max(1, len(samples) // n)
    subset = samples[::step][:n]  # spread over the whole index, all classes
    encode_samples(encoder, subset[:batch_size], device, batch_size, num_workers)  # warm-up
    if str(device).startswith("mps"):
        torch.mps.synchronize()
    start = time.perf_counter()
    encode_samples(encoder, subset, device, batch_size, num_workers)
    elapsed = time.perf_counter() - start
    per_image = elapsed / len(subset)
    return {
        "encoder": encoder.name,
        "device": str(device),
        "n_images": len(subset),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "seconds_total": elapsed,
        "seconds_per_image": per_image,
        "estimated_minutes": {k: v * per_image / 60 for k, v in totals.items()},
    }


def main() -> None:
    """Extract and cache frozen-encoder embeddings for the data splits (or run a timing test)."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--encoder", default=DEFAULT_ENCODER, choices=sorted(ENCODERS))
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--timing", type=int, metavar="N", help="only time N images")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--out-dir", type=Path, default=EMBEDDINGS_DIR)
    parser.add_argument("--device", help="override auto-detection, e.g. cpu")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else config.get_device()
    samples = build_index()
    splits = split_index(samples)
    paths = _paths(args.out_dir, args.encoder)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    encoder = load_encoder(args.encoder, device)
    print(f"Loaded {args.encoder} on {device} in {time.perf_counter() - start:.1f}s", flush=True)

    if args.timing:
        totals = {"all": len(samples), **{k: len(v) for k, v in splits.items()}}
        result = timing_test(
            encoder, samples, args.timing, device, args.batch_size, args.num_workers, totals
        )
        paths["timing"].write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        print(f"Wrote {paths['timing']}")
        return

    write_index(paths["index"], index_rows(splits))
    summary = {"encoder": args.encoder, "model_id": ENCODERS[args.encoder], "device": str(device)}
    summary["splits"] = {}
    for name in args.splits:
        t0 = time.perf_counter()
        emb, computed = extract_split(
            encoder,
            splits[name],
            name,
            args.out_dir,
            device,
            args.batch_size,
            args.num_workers,
            log=lambda msg: print(msg, flush=True),
        )
        seconds = time.perf_counter() - t0
        summary["splits"][name] = {
            "n": int(emb.shape[0]),
            "dim": int(emb.shape[1]),
            "computed_this_run": computed,
            "seconds_this_run": seconds,
            "finite": bool(np.isfinite(emb).all()),
        }
        print(f"{name}: {emb.shape} ({computed} computed this run, {seconds:.1f}s)", flush=True)
    paths["summary"].write_text(json.dumps(summary, indent=2))
    print(f"Wrote {paths['summary']} and {paths['index']}")


if __name__ == "__main__":
    main()
