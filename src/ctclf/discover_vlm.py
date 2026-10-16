"""Discover which NVIDIA Build VLMs accept CT slice images."""

import argparse
import json
import re
import time
from pathlib import Path

from ctclf import config
from ctclf.benchmark_vlm import PROMPT, parse_label
from ctclf.data import build_index, split_index
from ctclf.nvidia_client import NvidiaVLMClient

DISCOVERY_NAME = "vlm_discovery.json"
# From a public listing; any of these may be deprecated, gated or not free.
CANDIDATES: tuple[str, ...] = (
    "meta/llama-3.2-11b-vision-instruct",
    "meta/llama-3.2-90b-vision-instruct",
    "microsoft/phi-3-vision-128k-instruct",
    "nvidia/vila",
    "nvidia/neva-22b",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "meta/muse-glimmer-30b",
    "moonshotai/kimi-k3",
    "z-ai/glm-5.3-flash",
    "deepseek-ai/deepseek-v4.1-flash",
)
_VISION_HINT = re.compile(
    r"vision|vl\b|-vl|vila|neva|omni|multimodal|pixtral|paligemma|kosmos|fuyu|glimmer", re.I
)


def vision_like(model_ids: list[str]) -> list[str]:
    """Listed model ids whose name hints at image input (heuristic, for reporting only)."""
    return [m for m in model_ids if _VISION_HINT.search(m)]


def probe(client: NvidiaVLMClient, model: str, image: Path) -> dict:
    """Send one slice to model; report success, answer, parsed label and latency."""
    start = time.perf_counter()
    try:
        resp = client.ask(image, PROMPT, model=model, max_tokens=1024, use_cache=False)
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        return {
            "model": model,
            "ok": False,
            "status": status,
            "error": f"{type(exc).__name__}: {exc}"[:300],
            "elapsed_s": round(time.perf_counter() - start, 2),
        }
    return {
        "model": model,
        "ok": True,
        "answer": resp.text[:300],
        "parsed_label": parse_label(resp.text),
        "latency_s": round(resp.latency_s or 0.0, 2),
    }


def main() -> None:
    """List models, probe the candidates with one real test slice, save the results."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--extra", nargs="*", default=[], help="additional model ids to probe")
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    args = parser.parse_args()

    # Few retries: a missing or gated model fails fast with a 4xx anyway.
    client = NvidiaVLMClient(max_retries=2)
    listed = client.list_models()
    print(f"/models lists {len(listed)} models")

    test = split_index(build_index())["test"]
    image = next(s.path for s in test if config.CLASSES[s.label] == "pCT")
    print(f"Probe image: {image.name} (true label pCT)\n")

    results = []
    for model in [*CANDIDATES, *args.extra]:
        row = {"in_listing": model in listed, **probe(client, model, image)}
        results.append(row)
        if row["ok"]:
            print(
                f"OK   {model:48s} listed={row['in_listing']!s:5s} "
                f"{row['latency_s']:6.2f}s parsed={row['parsed_label']} "
                f"answer={row['answer'][:60]!r}"
            )
        else:
            print(f"FAIL {model:48s} listed={row['in_listing']!s:5s} {row['error'][:90]}")

    report = {
        "probe_image": image.name,
        "probe_true_label": "pCT",
        "listed_model_count": len(listed),
        "listed_vision_like": vision_like(listed),
        "results": results,
    }
    out = Path(args.results_dir) / DISCOVERY_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nVision-like ids in the listing: {report['listed_vision_like']}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
