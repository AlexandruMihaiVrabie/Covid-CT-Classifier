"""Tests for the NVIDIA client and VLM benchmark logic (mocked API, no network)."""

import io
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import openai
import pytest
import torch
from PIL import Image

from ctclf import benchmark_vlm as bench
from ctclf import config
from ctclf.data import Sample
from ctclf.discover_vlm import vision_like
from ctclf.model import build_model
from ctclf.nvidia_client import (
    NvidiaSettings,
    NvidiaVLMClient,
    RateLimiter,
    cache_key,
    encode_image,
    load_settings,
    strip_reasoning,
)

FAKE_KEY = "nvapi-test-secret-123"


class FakeAPI:
    """Minimal stand-in for openai.OpenAI: scripted answers or exceptions."""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.models = SimpleNamespace(
            list=lambda: SimpleNamespace(
                data=[SimpleNamespace(id="b/m"), SimpleNamespace(id="a/m")]
            )
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        message = SimpleNamespace(content=item)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _rate_limit_error(retry_after: str | None = None) -> openai.RateLimitError:
    headers = {"retry-after": retry_after} if retry_after else {}
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    response = httpx.Response(429, headers=headers, request=request)
    return openai.RateLimitError("rate limited", response=response, body=None)


def _client(api: FakeAPI, tmp_path: Path, sleeps: list[float]) -> NvidiaVLMClient:
    settings = NvidiaSettings(api_key=FAKE_KEY, model="vendor/vlm")
    return NvidiaVLMClient(
        settings, cache_dir=tmp_path / "cache", client=api, sleep=sleeps.append,
        requests_per_minute=6000,
    )  # fmt: skip


@pytest.fixture
def image() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (600, 700), dtype=np.uint8), mode="L")


def test_settings_hide_key_and_detect_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    s = NvidiaSettings(api_key=FAKE_KEY, model="m")
    assert FAKE_KEY not in repr(s)
    assert s.has_key
    assert not NvidiaSettings(api_key="nvapi-your-key-here").has_key
    assert not NvidiaSettings(api_key=None).has_key
    for var in ("NVIDIA_API_KEY", "NVIDIA_BASE_URL", "NVIDIA_VLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    empty = load_settings(env_file=None)
    assert not empty.has_key and empty.model is None
    assert empty.base_url.startswith("https://integrate.api.nvidia.com")


def test_client_without_key_raises() -> None:
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        NvidiaVLMClient(NvidiaSettings(api_key=None))


def test_encode_image_is_small_jpeg_and_deterministic(image: Image.Image) -> None:
    data = encode_image(image)
    assert data[:2] == b"\xff\xd8"
    assert max(Image.open(io.BytesIO(data)).size) == 512
    assert encode_image(image) == data
    assert cache_key("m", data, "p") != cache_key("m2", data, "p")
    assert cache_key("m", data, "p") != cache_key("m", data, "p2")


def test_ask_sends_image_and_caches(image: Image.Image, tmp_path: Path) -> None:
    api, sleeps = FakeAPI(["pCT"]), []
    client = _client(api, tmp_path, sleeps)
    first = client.ask(image, "prompt")
    second = client.ask(image, "prompt")
    assert (first.text, first.cached) == ("pCT", False)
    assert (second.text, second.cached) == ("pCT", True)
    assert len(api.calls) == 1
    content = api.calls[0]["messages"][0]["content"]
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert api.calls[0]["model"] == "vendor/vlm"
    cached_files = list((tmp_path / "cache").glob("*.json"))
    assert len(cached_files) == 1
    assert FAKE_KEY not in cached_files[0].read_text()


def test_retry_on_rate_limit_then_success(image: Image.Image, tmp_path: Path) -> None:
    api, sleeps = FakeAPI([_rate_limit_error(), _rate_limit_error("7"), "NiCT"]), []
    client = _client(api, tmp_path, sleeps)
    assert client.ask(image, "p").text == "NiCT"
    assert len(api.calls) == 3
    backoffs = [s for s in sleeps if s >= 1]
    assert backoffs == [2.0, 7.0]  # exponential first, then the server's Retry-After


def test_retry_gives_up_and_non_retryable_fails_fast(image: Image.Image, tmp_path: Path) -> None:
    api = FakeAPI([_rate_limit_error()] * 3)
    client = _client(api, tmp_path, [])
    client.max_retries = 2
    with pytest.raises(openai.RateLimitError):
        client.ask(image, "p")
    assert len(api.calls) == 3

    request = httpx.Request("POST", "https://example.invalid")
    not_found = openai.NotFoundError(
        "no such model", response=httpx.Response(404, request=request), body=None
    )
    api2 = FakeAPI([not_found])
    with pytest.raises(openai.NotFoundError):
        _client(api2, tmp_path, []).ask(image, "p")
    assert len(api2.calls) == 1


def test_list_models_sorted(tmp_path: Path) -> None:
    assert _client(FakeAPI([]), tmp_path, []).list_models() == ["a/m", "b/m"]


def test_rate_limiter_spaces_calls() -> None:
    now = [0.0]
    sleeps: list[float] = []

    def sleep(s: float) -> None:
        sleeps.append(s)
        now[0] += s

    limiter = RateLimiter(30, clock=lambda: now[0], sleep=sleep)
    limiter.wait()
    now[0] += 0.5
    limiter.wait()
    now[0] += 5.0
    limiter.wait()
    assert sleeps == [pytest.approx(1.5)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("pCT", "pCT"),
        (" nct. ", "nCT"),
        ("**NiCT**", "NiCT"),
        ("The answer is pCT.", "pCT"),
        ("<think>could be nCT or NiCT</think>\npCT", "pCT"),
        ("nCT or pCT", None),
        ("COVID-19 positive", None),
        ("", None),
    ],
)
def test_parse_label(text: str, expected: str | None) -> None:
    assert bench.parse_label(text) == expected


def test_strip_reasoning_unclosed() -> None:
    assert strip_reasoning("<think>still thinking") == ""


def test_wilson_interval_known_values() -> None:
    lo, hi = bench.wilson_interval(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3)
    assert hi == pytest.approx(0.9433, abs=1e-3)
    assert bench.wilson_interval(0, 0) == (0.0, 0.0)
    assert bench.wilson_interval(10, 10)[1] == 1.0


def test_stratified_sample_quota_and_determinism(tmp_path: Path) -> None:
    samples = [Sample(tmp_path / f"{c}{i}.jpg", lbl) for lbl, c in enumerate(config.CLASSES)
               for i in range(60)]  # fmt: skip
    assert bench.per_class_quota(100) == [34, 33, 33]
    a = bench.stratified_sample(samples)
    b = bench.stratified_sample(list(reversed(samples)))
    assert a == b
    assert [sum(s.label == i for s in a) for i in range(3)] == [34, 33, 33]
    assert len({s.path for s in a}) == 100


def _rows() -> list[dict]:
    # true, cnn, vlm (None = unparseable)
    spec = [(0, 0, "nCT"), (0, 0, None), (1, 1, "pCT"), (1, 0, "nCT"), (2, 2, "NiCT"), (2, 2, None)]
    return [
        {
            "true_index": t,
            "true_label": config.CLASSES[t],
            "cnn_index": c,
            "vlm_label": v,
            "error": None,
        }
        for t, c, v in spec
    ]


def test_summarize_counts_unparseable_as_wrong() -> None:
    s = bench.summarize(_rows(), "vendor/vlm")
    assert s["n"] == 6
    assert s["cnn"]["correct"] == 5
    assert s["vlm"]["correct"] == 3
    assert s["vlm"]["accuracy"] == pytest.approx(0.5)
    assert s["vlm_unparseable"] == 2
    assert s["vlm_unparseable_per_class"] == {"nCT": 1, "pCT": 0, "NiCT": 1}
    assert s["vlm"]["per_class"]["NiCT"]["recall"] == pytest.approx(0.5)
    assert s["vlm"]["per_class"]["NiCT"]["recall_ci95"] == pytest.approx(
        bench.wilson_interval(1, 2)
    )


def test_run_benchmark_and_check_summary(tmp_path: Path, image: Image.Image) -> None:
    samples = []
    for label, cls in enumerate(config.CLASSES):
        (tmp_path / cls).mkdir()
        path = tmp_path / cls / f"{cls}1.jpg"
        # Distinct images, otherwise the cache would answer for the later ones.
        Image.eval(image, lambda px, k=label: (px + 60 * k) % 256).save(path)
        samples.append(Sample(path, label))
    api = FakeAPI(["nCT", "I am not sure", _rate_limit_error()])
    client = _client(api, tmp_path, [])
    client.max_retries = 0
    torch.manual_seed(0)
    rows = bench.run_benchmark(client, "vendor/vlm", samples, build_model(pretrained=False).eval())
    assert [r["vlm_label"] for r in rows] == ["nCT", None, None]
    assert rows[2]["error"].startswith("RateLimitError")
    assert FAKE_KEY not in json.dumps(rows)

    summary = bench.summarize(rows, "vendor/vlm")
    assert summary["vlm_unparseable"] == 2 and summary["vlm_errors"] == 1
    raw = {"vlm_model": "vendor/vlm", "rows": rows}
    (tmp_path / bench.RAW_NAME).write_text(json.dumps(raw))
    (tmp_path / bench.SUMMARY_NAME).write_text(json.dumps(summary))
    assert bench.check_summary(tmp_path)
    summary["vlm"]["accuracy"] = 0.99
    (tmp_path / bench.SUMMARY_NAME).write_text(json.dumps(summary))
    assert not bench.check_summary(tmp_path)


def test_vision_like_filter() -> None:
    ids = ["meta/llama-3.2-11b-vision-instruct", "meta/llama-3.1-8b-instruct", "nvidia/vila"]
    assert vision_like(ids) == ["meta/llama-3.2-11b-vision-instruct", "nvidia/vila"]
