"""Zero-shot CNN-vs-VLM benchmark on a stratified test sample."""

import argparse
import json
import math
import random
import re
from pathlib import Path

import torch

from ctclf import config
from ctclf.data import Sample, build_index, load_image, split_index
from ctclf.evaluate import classification_metrics
from ctclf.model import load_model
from ctclf.nvidia_client import NvidiaVLMClient, strip_reasoning
from ctclf.predict import DEFAULT_CHECKPOINT, predict_image

RAW_NAME = "vlm_benchmark_raw.json"
SUMMARY_NAME = "vlm_benchmark_summary.json"
SAMPLE_SIZE = 100
UNPARSEABLE = -1

PROMPT = """You are shown one axial chest CT slice. Classify it into exactly one of these classes:
nCT: normal. The lungs are visible and show no signs of COVID-19 pneumonia.
pCT: COVID-19 positive. The lungs are visible and show findings consistent with COVID-19 \
pneumonia, such as ground-glass opacities or consolidation.
NiCT: non-informative. The slice does not show enough lung tissue to judge, for example \
it is at the level of the neck or the upper abdomen.
Answer with exactly one label: nCT, pCT or NiCT. Do not write anything else."""

_LABEL_RE = re.compile(r"\b(nCT|pCT|NiCT)\b", re.IGNORECASE)
_CANONICAL = {cls.lower(): cls for cls in config.CLASSES}


def parse_label(text: str) -> str | None:
    """Return the single class label in the answer, or None if absent or ambiguous."""
    answer = strip_reasoning(text).strip().strip(".*`'\" ")
    if answer.lower() in _CANONICAL:
        return _CANONICAL[answer.lower()]
    found = {_CANONICAL[m.lower()] for m in _LABEL_RE.findall(answer)}
    return found.pop() if len(found) == 1 else None


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def per_class_quota(n: int, n_classes: int = len(config.CLASSES)) -> list[int]:
    """Split n as evenly as possible, extra items going to the first classes (34/33/33)."""
    return [n // n_classes + (i < n % n_classes) for i in range(n_classes)]


def stratified_sample(
    samples: list[Sample], n: int = SAMPLE_SIZE, seed: int = config.SEED
) -> list[Sample]:
    """Deterministic per-class random sample with quotas from per_class_quota."""
    rng = random.Random(seed)
    picked: list[Sample] = []
    for label, quota in enumerate(per_class_quota(n)):
        group = sorted((s for s in samples if s.label == label), key=lambda s: s.path.name)
        picked.extend(rng.sample(group, quota))
    return picked


def score(targets: list[int], preds: list[int]) -> dict:
    """Accuracy with Wilson CI, per-class recall with Wilson CI and sklearn metrics."""
    n = len(targets)
    correct = sum(t == p for t, p in zip(targets, preds, strict=True))
    metrics = classification_metrics(targets, preds)
    for label, cls in enumerate(config.CLASSES):
        idx = [i for i, t in enumerate(targets) if t == label]
        hits = sum(preds[i] == label for i in idx)
        metrics["per_class"][cls]["correct"] = hits
        metrics["per_class"][cls]["recall_ci95"] = wilson_interval(hits, len(idx))
    metrics["correct"] = correct
    metrics["accuracy_ci95"] = wilson_interval(correct, n)
    return metrics


def summarize(rows: list[dict], model: str) -> dict:
    """Summary metrics for CNN and VLM, recomputed from the raw rows."""
    targets = [r["true_index"] for r in rows]
    vlm_preds = [
        config.CLASSES.index(r["vlm_label"]) if r["vlm_label"] else UNPARSEABLE for r in rows
    ]
    unparseable = [r for r in rows if r["vlm_label"] is None]
    return {
        "vlm_model": model,
        "n": len(rows),
        "per_class_n": {cls: sum(t == i for t in targets) for i, cls in enumerate(config.CLASSES)},
        "note": "Small sample: confidence intervals are wide. Unparseable VLM answers "
        "count as wrong.",
        "cnn": score(targets, [r["cnn_index"] for r in rows]),
        "vlm": score(targets, vlm_preds),
        "vlm_unparseable": len(unparseable),
        "vlm_unparseable_per_class": {
            cls: sum(r["true_label"] == cls for r in unparseable) for cls in config.CLASSES
        },
        "vlm_errors": sum(bool(r.get("error")) for r in rows),
    }


def run_benchmark(
    client: NvidiaVLMClient,
    model_name: str,
    samples: list[Sample],
    cnn: torch.nn.Module,
    device: torch.device | str = "cpu",
) -> list[dict]:
    """Classify each sample with the CNN and the VLM; one raw row per image."""
    rows = []
    for i, s in enumerate(samples, 1):
        image = load_image(s.path)
        cnn_out = predict_image(cnn, image, device)
        row = {
            "image": s.path.name,
            "true_label": config.CLASSES[s.label],
            "true_index": s.label,
            "cnn_label": cnn_out["label"],
            "cnn_index": cnn_out["label_index"],
            "cnn_probabilities": cnn_out["probabilities"],
            "vlm_raw": None,
            "vlm_label": None,
            "vlm_cached": None,
            "vlm_latency_s": None,
            "error": None,
        }
        try:
            resp = client.ask(s.path, PROMPT, model=model_name, max_tokens=1024)
            row.update(
                vlm_raw=resp.text,
                vlm_label=parse_label(resp.text),
                vlm_cached=resp.cached,
                vlm_latency_s=resp.latency_s,
            )
        except Exception as exc:  # keep going; counted as wrong and reported
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
        rows.append(row)
        print(
            f"[{i:3d}/{len(samples)}] {s.path.name:14s} true {row['true_label']:5s} "
            f"cnn {row['cnn_label']:5s} vlm {row['vlm_label'] or '-':5s}"
            f"{' (cached)' if row['vlm_cached'] else ''}{' ERROR' if row['error'] else ''}",
            flush=True,
        )
    return rows


def check_summary(results_dir: Path) -> bool:
    """Recompute the summary from the raw file and compare with the saved summary."""
    raw = json.loads((results_dir / RAW_NAME).read_text())
    saved = json.loads((results_dir / SUMMARY_NAME).read_text())
    recomputed = json.loads(json.dumps(summarize(raw["rows"], raw["vlm_model"])))
    return recomputed == saved


def _fmt(m: dict) -> str:
    lo, hi = m["accuracy_ci95"]
    return f"acc {m['accuracy']:.3f} [{lo:.3f}, {hi:.3f}] ({m['correct']}/{m['n']})"


def main() -> None:
    """Run the CNN-vs-VLM benchmark and write raw results and a summary."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--model", help="VLM id (default: NVIDIA_VLM_MODEL)")
    parser.add_argument("--n", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--device", help="override auto-detection, e.g. cpu")
    parser.add_argument("--check", action="store_true", help="only verify summary vs raw")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    if args.check:
        ok = check_summary(results_dir)
        print(f"Summary matches raw file: {'PASS' if ok else 'FAIL'}")
        raise SystemExit(0 if ok else 1)

    client = NvidiaVLMClient()
    model_name = args.model or client.settings.model
    if not model_name:
        parser.error("set NVIDIA_VLM_MODEL or pass --model")
    device = torch.device(args.device) if args.device else config.get_device()
    cnn = load_model(args.checkpoint, device)
    samples = stratified_sample(split_index(build_index())["test"], args.n)

    rows = run_benchmark(client, model_name, samples, cnn, device)
    summary = summarize(rows, model_name)
    results_dir.mkdir(parents=True, exist_ok=True)
    raw = {"vlm_model": model_name, "prompt": PROMPT, "seed": config.SEED, "rows": rows}
    (results_dir / RAW_NAME).write_text(json.dumps(raw, indent=2))
    (results_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2))

    print(f"\nModel: {model_name}, n={summary['n']} {summary['per_class_n']}")
    print(f"CNN: {_fmt(summary['cnn'])}  macro-F1 {summary['cnn']['macro_f1']:.3f}")
    print(f"VLM: {_fmt(summary['vlm'])}  macro-F1 {summary['vlm']['macro_f1']:.3f}")
    print(
        f"VLM unparseable: {summary['vlm_unparseable']} {summary['vlm_unparseable_per_class']}, "
        f"errors: {summary['vlm_errors']}"
    )
    for cls in config.CLASSES:
        c, v = summary["cnn"]["per_class"][cls], summary["vlm"]["per_class"][cls]
        (c_lo, c_hi), (v_lo, v_hi) = c["recall_ci95"], v["recall_ci95"]
        print(
            f"  {cls:5s} recall CNN {c['recall']:.3f} [{c_lo:.3f}, {c_hi:.3f}]"
            f"  VLM {v['recall']:.3f} [{v_lo:.3f}, {v_hi:.3f}]"
        )
    print(f"Wrote {results_dir / RAW_NAME} and {results_dir / SUMMARY_NAME}")
    print(f"Summary matches raw file: {'PASS' if check_summary(results_dir) else 'FAIL'}")


if __name__ == "__main__":
    main()
