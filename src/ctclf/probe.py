"""Linear probe and zero-shot on frozen-encoder embeddings, plus the model comparison."""

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from ctclf import config
from ctclf.benchmark_vlm import RAW_NAME, SUMMARY_NAME, score, wilson_interval
from ctclf.data import Sample, build_datasets
from ctclf.embeddings import (
    DEFAULT_ENCODER,
    EMBEDDINGS_DIR,
    ENCODERS,
    cached_text_embeddings,
    load_split_embeddings,
)
from ctclf.evaluate import METRICS_NAME, predict_loader
from ctclf.model import load_model
from ctclf.predict import DEFAULT_CHECKPOINT
from ctclf.train import make_loader

PROBE_SUMMARY_NAME = "probe_summary.json"
COMPARISON_NAME = "comparison.json"
COMPARISON_MD_NAME = "comparison.md"
PREDICTIONS_NAME = "test_predictions.csv"

C_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
CLASS_WEIGHTS: tuple[str | None, ...] = (None, "balanced")

# Fixed before looking at any result; one set is picked on validation macro-F1 only.
# Several prompts per class are averaged (prompt ensembling).
PROMPT_SETS: dict[str, dict[str, list[str]]] = {
    "short": {
        "nCT": ["a normal chest CT scan"],
        "pCT": ["a chest CT scan with COVID-19 pneumonia"],
        "NiCT": ["a CT slice without lung tissue"],
    },
    "biomedclip_template": {
        "nCT": ["this is a photo of a normal chest CT scan"],
        "pCT": ["this is a photo of a chest CT scan showing COVID-19 pneumonia"],
        "NiCT": ["this is a photo of a CT slice that does not show the lungs"],
    },
    "descriptive": {
        "nCT": [
            "an axial chest CT slice showing normal healthy lungs",
            "a chest CT image of clear lungs with no opacities",
        ],
        "pCT": [
            "an axial chest CT slice showing ground-glass opacities of COVID-19 pneumonia",
            "a chest CT image of lungs with bilateral consolidation from COVID-19",
        ],
        "NiCT": [
            "an axial CT slice of the neck or upper abdomen with no lungs visible",
            "a non-informative CT image without lung parenchyma",
        ],
    },
}


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation (zero rows stay zero)."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1.0, norms)


def fit_probe(
    x: np.ndarray, y: Sequence[int], c: float, class_weight: str | None
) -> LogisticRegression:
    """Multinomial logistic regression on L2-normalised embeddings."""
    clf = LogisticRegression(C=c, class_weight=class_weight, max_iter=5000)
    return clf.fit(l2_normalize(x), list(y))


def predict_probe(clf: LogisticRegression, x: np.ndarray) -> list[int]:
    """Class indices predicted by a fitted probe."""
    return clf.predict(l2_normalize(x)).astype(int).tolist()


def select_probe(
    x_train: np.ndarray,
    y_train: Sequence[int],
    x_val: np.ndarray,
    y_val: Sequence[int],
    c_grid: Sequence[float] = C_GRID,
    class_weights: Sequence[str | None] = CLASS_WEIGHTS,
) -> tuple[dict, list[dict]]:
    """Grid search on validation macro-F1 (ties: accuracy, then smaller C); test is not used."""
    grid = []
    for cw in class_weights:
        for c in c_grid:
            m = score(list(y_val), predict_probe(fit_probe(x_train, y_train, c, cw), x_val))
            grid.append(
                {
                    "C": c,
                    "class_weight": cw,
                    "val_macro_f1": m["macro_f1"],
                    "val_accuracy": m["accuracy"],
                }
            )
    best = max(grid, key=lambda r: (r["val_macro_f1"], r["val_accuracy"], -r["C"]))
    return {"C": best["C"], "class_weight": best["class_weight"]}, grid


def class_text_embeddings(
    text_emb: np.ndarray, prompt_labels: Sequence[int], n_classes: int = len(config.CLASSES)
) -> np.ndarray:
    """Average the normalised prompt embeddings of each class and renormalise: (n_classes, dim)."""
    text_emb = l2_normalize(text_emb)
    labels = np.asarray(prompt_labels)
    return l2_normalize(np.stack([text_emb[labels == k].mean(0) for k in range(n_classes)]))


def zero_shot_predict(image_emb: np.ndarray, class_emb: np.ndarray) -> list[int]:
    """Class with the highest cosine similarity between image and class text embedding."""
    return (l2_normalize(image_emb) @ class_emb.T).argmax(1).astype(int).tolist()


def flatten_prompts(prompt_set: dict[str, list[str]]) -> tuple[list[str], list[int]]:
    """Prompts in class order with their class indices."""
    prompts, labels = [], []
    for k, cls in enumerate(config.CLASSES):
        prompts.extend(prompt_set[cls])
        labels.extend([k] * len(prompt_set[cls]))
    return prompts, labels


def mcnemar_exact(ref_correct: Sequence[bool], other_correct: Sequence[bool]) -> dict:
    """Exact (binomial) two-sided McNemar test on paired correctness."""
    b = sum(r and not o for r, o in zip(ref_correct, other_correct, strict=True))
    c = sum(o and not r for r, o in zip(ref_correct, other_correct, strict=True))
    n = b + c
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1))
    p = min(1.0, 2 * tail / 2**n) if n else 1.0
    return {
        "n_pairs": len(ref_correct),
        "ref_only_correct": b,
        "other_only_correct": c,
        "p_value": p,
    }


def with_cis(metrics: dict) -> dict:
    """Add correct counts and Wilson 95% CIs to metrics that only carry a confusion matrix."""
    cm = metrics["confusion_matrix"]
    out = json.loads(json.dumps(metrics))
    out["correct"] = sum(cm[i][i] for i in range(len(cm)))
    out["accuracy_ci95"] = wilson_interval(out["correct"], out["n"])
    for i, cls in enumerate(config.CLASSES):
        out["per_class"][cls]["correct"] = cm[i][i]
        out["per_class"][cls]["recall_ci95"] = wilson_interval(cm[i][i], sum(cm[i]))
    return out


def file_hash(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def duplicated_in_train(train: list[Sample], test: list[Sample]) -> list[str]:
    """Test filenames whose bytes are identical to some training image."""
    train_hashes = {file_hash(s.path) for s in train}
    return [s.path.name for s in test if file_hash(s.path) in train_hashes]


def cnn_predictions(checkpoint: Path, dataset, device: torch.device) -> list[int]:
    """Per-image CNN predictions over a dataset (eval transforms), in dataset order."""
    model = load_model(checkpoint, device)
    loader = make_loader(dataset, config.BATCH_SIZE, False, 0)
    _, preds, _ = predict_loader(model, loader, device)
    return preds


def cnn_recheck(recomputed: dict, saved: dict, tol: float = 1e-9) -> dict:
    """Compare CNN test metrics recomputed per image with those in metrics.json."""
    diffs = {k: abs(recomputed[k] - saved[k]) for k in ("accuracy", "macro_f1")}
    same_cm = recomputed["confusion_matrix"] == saved["confusion_matrix"]
    return {
        "abs_diff": diffs,
        "confusion_matrix_identical": same_cm,
        "passed": same_cm and all(d <= tol for d in diffs.values()),
    }


def _compact(m: dict, source: str) -> dict:
    return {
        "source": source,
        "n": m["n"],
        "correct": m["correct"],
        "accuracy": m["accuracy"],
        "accuracy_ci95": m["accuracy_ci95"],
        "macro_f1": m["macro_f1"],
        "recall": {cls: m["per_class"][cls]["recall"] for cls in config.CLASSES},
        "recall_ci95": {cls: m["per_class"][cls]["recall_ci95"] for cls in config.CLASSES},
        "confusion_matrix": m["confusion_matrix"],
    }


def build_comparison(probe_summary: dict, metrics: dict, vlm_summary: dict) -> dict:
    """Four-way (100-slice) and three-way (full test) comparison from the result files."""
    enc = probe_summary["encoder"]
    sub = probe_summary["subset_100"]
    return {
        "note": "CNN and VLM numbers are read from metrics.json and vlm_benchmark_summary.json; "
        "probe and zero-shot from probe_summary.json. Per-slice split: test accuracy is "
        "optimistic. Educational only, not a medical device.",
        "subset_100": {
            "n": vlm_summary["n"],
            "per_class_n": vlm_summary["per_class_n"],
            "models": {
                "cnn": _compact(vlm_summary["cnn"], SUMMARY_NAME),
                f"{enc}_probe": _compact(sub["probe"], PROBE_SUMMARY_NAME),
                f"{enc}_zero_shot": _compact(sub["zero_shot"], PROBE_SUMMARY_NAME),
                "nvidia_vlm_zero_shot": {
                    **_compact(vlm_summary["vlm"], SUMMARY_NAME),
                    "model": vlm_summary["vlm_model"],
                    "unparseable_counted_wrong": vlm_summary["vlm_unparseable"],
                },
            },
        },
        "full_test": {
            "n": metrics["splits"]["test"]["n"],
            "models": {
                "cnn": _compact(with_cis(metrics["splits"]["test"]), METRICS_NAME),
                f"{enc}_probe": _compact(probe_summary["probe"]["test"], PROBE_SUMMARY_NAME),
                f"{enc}_zero_shot": _compact(
                    probe_summary["zero_shot"]["test"], PROBE_SUMMARY_NAME
                ),
            },
            "mcnemar_cnn_vs_probe": probe_summary["mcnemar_cnn_vs_probe_full_test"],
        },
        "duplicate_sensitivity": probe_summary["duplicate_sensitivity"],
    }


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _table(models: dict) -> list[str]:
    head = "| Model | Accuracy [95% CI] | Macro-F1 | Recall nCT | Recall pCT | Recall NiCT |"
    lines = [head, "|---|---|---|---|---|---|"]
    for name, m in models.items():
        lo, hi = m["accuracy_ci95"]
        rec = " | ".join(_pct(m["recall"][c]) for c in config.CLASSES)
        lines.append(
            f"| {name} | {_pct(m['accuracy'])} [{_pct(lo)}, {_pct(hi)}] ({m['correct']}/{m['n']}) "
            f"| {m['macro_f1']:.3f} | {rec} |"
        )
    return lines


def comparison_markdown(comp: dict) -> str:
    """Human-readable comparison tables."""
    sub, full = comp["subset_100"], comp["full_test"]
    mc = full["mcnemar_cnn_vs_probe"]
    dup = comp["duplicate_sensitivity"]
    lines = [
        "# Model comparison",
        "",
        "Generated by `python -m ctclf.probe`. " + comp["note"],
        "",
        f"## Same {sub['n']} test slices as the NVIDIA VLM benchmark {sub['per_class_n']}",
        "",
        *_table(sub["models"]),
        "",
        "Unparseable VLM answers count as wrong "
        f"({sub['models']['nvidia_vlm_zero_shot']['unparseable_counted_wrong']}).",
        "",
        f"## Full test split (n = {full['n']})",
        "",
        *_table(full["models"]),
        "",
        f"Exact McNemar, CNN vs probe (paired, full test): CNN-only correct "
        f"{mc['ref_only_correct']}, probe-only correct {mc['other_only_correct']}, "
        f"p = {mc['p_value']:.3g}.",
        "",
        "## Duplicate sensitivity check",
        "",
        f"{dup['test_images_with_exact_duplicate_in_train']} test images have a byte-identical "
        f"copy in train. Without them (n = {dup['n_without']}): CNN accuracy "
        f"{_pct(dup['cnn_without']['accuracy'])} (macro-F1 {dup['cnn_without']['macro_f1']:.3f}), "
        f"probe accuracy {_pct(dup['probe_without']['accuracy'])} "
        f"(macro-F1 {dup['probe_without']['macro_f1']:.3f}). Report only; the split is unchanged.",
        "",
    ]
    return "\n".join(lines)


def check_files(results_dir: Path) -> bool:
    """Rebuild comparison.json from its source files and compare with the saved one."""
    load = lambda name: json.loads((results_dir / name).read_text())  # noqa: E731
    rebuilt = build_comparison(load(PROBE_SUMMARY_NAME), load(METRICS_NAME), load(SUMMARY_NAME))
    return json.loads(json.dumps(rebuilt)) == load(COMPARISON_NAME)


def main() -> None:
    """Fit the probe and zero-shot on val, score test once, write summary and comparison."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--encoder", default=DEFAULT_ENCODER, choices=sorted(ENCODERS))
    parser.add_argument("--embeddings-dir", type=Path, default=EMBEDDINGS_DIR)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", help="override auto-detection, e.g. cpu")
    parser.add_argument("--check", action="store_true", help="only verify comparison vs sources")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    if args.check:
        ok = check_files(results_dir)
        print(f"comparison.json matches its source files: {'PASS' if ok else 'FAIL'}")
        raise SystemExit(0 if ok else 1)

    device = torch.device(args.device) if args.device else config.get_device()
    datasets = build_datasets()
    splits = {name: ds.samples for name, ds in datasets.items()}
    emb, y = {}, {}
    for name, samples in splits.items():
        emb[name], _ = load_split_embeddings(name, args.encoder, args.embeddings_dir, samples)
        y[name] = [s.label for s in samples]
    print(f"Embeddings: { {k: v.shape for k, v in emb.items()} }", flush=True)

    # Linear probe: hyperparameters on val only, then a single test evaluation.
    params, grid = select_probe(emb["train"], y["train"], emb["val"], y["val"])
    clf = fit_probe(emb["train"], y["train"], params["C"], params["class_weight"])
    probe_val = score(y["val"], predict_probe(clf, emb["val"]))
    for row in grid:
        print(
            f"  probe C={row['C']:<6} cw={row['class_weight']!s:8s} val macro-F1 "
            f"{row['val_macro_f1']:.4f} acc {row['val_accuracy']:.4f}"
        )
    print(f"Selected probe {params} (val macro-F1 {probe_val['macro_f1']:.4f})", flush=True)

    # Zero-shot: prompt set chosen on val only.
    zs_val, class_embs = {}, {}
    for set_name, prompt_set in PROMPT_SETS.items():
        prompts, labels = flatten_prompts(prompt_set)
        text = cached_text_embeddings(prompts, args.encoder, args.embeddings_dir, device=device)
        class_embs[set_name] = class_text_embeddings(text, labels)
        zs_val[set_name] = score(y["val"], zero_shot_predict(emb["val"], class_embs[set_name]))
        m = zs_val[set_name]
        print(f"  zero-shot '{set_name}' val macro-F1 {m['macro_f1']:.4f} acc {m['accuracy']:.4f}")
    best_set = max(zs_val, key=lambda k: (zs_val[k]["macro_f1"], zs_val[k]["accuracy"]))
    print(f"Selected prompt set '{best_set}'", flush=True)

    # Test split: scored once with the selected settings.
    probe_test_preds = predict_probe(clf, emb["test"])
    zs_test_preds = zero_shot_predict(emb["test"], class_embs[best_set])
    probe_test = score(y["test"], probe_test_preds)
    zs_test = score(y["test"], zs_test_preds)

    # CNN per-image predictions on the full test split (metrics.json has aggregates only).
    metrics = json.loads((results_dir / METRICS_NAME).read_text())
    cnn_test_preds = cnn_predictions(args.checkpoint, datasets["test"], device)
    cnn_test = score(y["test"], cnn_test_preds)
    recheck = cnn_recheck(cnn_test, metrics["splits"]["test"])
    print(f"CNN recomputed vs metrics.json: {'PASS' if recheck['passed'] else 'FAIL'}", flush=True)

    correct = lambda preds: [p == t for p, t in zip(preds, y["test"], strict=True)]  # noqa: E731
    mcnemar = mcnemar_exact(correct(cnn_test_preds), correct(probe_test_preds))

    # Same 100 slices as the VLM benchmark.
    vlm_raw = json.loads((results_dir / RAW_NAME).read_text())
    vlm_summary = json.loads((results_dir / SUMMARY_NAME).read_text())
    pos = {s.path.name: i for i, s in enumerate(splits["test"])}
    sub_idx = [pos[r["image"]] for r in vlm_raw["rows"]]
    sub_y = [r["true_index"] for r in vlm_raw["rows"]]
    if sub_y != [y["test"][i] for i in sub_idx]:
        raise ValueError("VLM benchmark labels do not match the test split")
    cnn_sub_match = [cnn_test_preds[i] for i in sub_idx] == [
        r["cnn_index"] for r in vlm_raw["rows"]
    ]
    subset = {
        "images": [r["image"] for r in vlm_raw["rows"]],
        "probe": score(sub_y, [probe_test_preds[i] for i in sub_idx]),
        "zero_shot": score(sub_y, [zs_test_preds[i] for i in sub_idx]),
        "cnn_rerun_matches_vlm_raw": cnn_sub_match,
    }

    # Sensitivity: drop test images that have a byte-identical copy in train.
    dup_names = set(duplicated_in_train(splits["train"], splits["test"]))
    keep = [i for i, s in enumerate(splits["test"]) if s.path.name not in dup_names]
    keep_y = [y["test"][i] for i in keep]
    dup = {
        "data_report_duplicates_across_splits": json.loads(
            (results_dir / "data_report.json").read_text()
        )["duplicates_across_splits"],
        "test_images_with_exact_duplicate_in_train": len(dup_names),
        "test_images": sorted(dup_names),
        "n_without": len(keep),
        "cnn_without": score(keep_y, [cnn_test_preds[i] for i in keep]),
        "probe_without": score(keep_y, [probe_test_preds[i] for i in keep]),
    }

    summary = {
        "encoder": args.encoder,
        "model_id": ENCODERS[args.encoder],
        "embedding_dim": int(emb["train"].shape[1]),
        "n": {k: len(v) for k, v in y.items()},
        "protocol": "L2-normalised embeddings; probe hyperparameters and zero-shot prompt set "
        "selected on val macro-F1 only; probe fit on train; test scored once.",
        "probe": {
            "grid": grid,
            "selected": params,
            "val": probe_val,
            "test": probe_test,
        },
        "zero_shot": {
            "prompt_sets": PROMPT_SETS,
            "val_by_prompt_set": zs_val,
            "selected_prompt_set": best_set,
            "test": zs_test,
        },
        "cnn_test_recheck_vs_metrics_json": recheck,
        "mcnemar_cnn_vs_probe_full_test": mcnemar,
        "subset_100": subset,
        "duplicate_sensitivity": dup,
    }
    comparison = build_comparison(summary, metrics, vlm_summary)

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / PROBE_SUMMARY_NAME).write_text(json.dumps(summary, indent=2))
    (results_dir / COMPARISON_NAME).write_text(json.dumps(comparison, indent=2))
    (results_dir / COMPARISON_MD_NAME).write_text(comparison_markdown(comparison))
    with (results_dir / PREDICTIONS_NAME).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "true", "cnn", "probe", "zero_shot"])
        for i, s in enumerate(splits["test"]):
            cls = config.CLASSES
            writer.writerow(
                [s.path.name, cls[y["test"][i]], cls[cnn_test_preds[i]],
                 cls[probe_test_preds[i]], cls[zs_test_preds[i]]]
            )  # fmt: skip

    print()
    print(comparison_markdown(comparison))
    print(f"CNN on the 100 slices matches vlm_benchmark_raw.json: {cnn_sub_match}")
    print(
        f"Wrote {PROBE_SUMMARY_NAME}, {COMPARISON_NAME}, {COMPARISON_MD_NAME}, {PREDICTIONS_NAME}"
    )
    ok = check_files(results_dir)
    print(f"comparison.json matches its source files: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
