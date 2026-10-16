"""Throttled, cached client for NVIDIA's OpenAI-compatible API."""

import base64
import hashlib
import io
import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import openai
from dotenv import load_dotenv
from PIL import Image

from ctclf import config

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
PLACEHOLDER_KEY = "nvapi-your-key-here"
CACHE_DIR = config.RESULTS_DIR / "vlm_cache"
# The free tier is said to allow ~40 requests/minute (unverified); stay well below it.
DEFAULT_REQUESTS_PER_MINUTE = 30
MAX_IMAGE_SIDE = 512

RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    openai.RateLimitError,
    openai.APIConnectionError,  # includes APITimeoutError
    openai.InternalServerError,
)


@dataclass(frozen=True)
class NvidiaSettings:
    """Endpoint settings; the key is excluded from repr so it never ends up in logs."""

    api_key: str | None = field(repr=False)
    base_url: str = DEFAULT_BASE_URL
    model: str | None = None

    @property
    def has_key(self) -> bool:
        """True when a real (non-placeholder) key is configured."""
        return bool(self.api_key) and self.api_key != PLACEHOLDER_KEY


def load_settings(env_file: Path | None = config.PROJECT_ROOT / ".env") -> NvidiaSettings:
    """Read NVIDIA_API_KEY, NVIDIA_BASE_URL and NVIDIA_VLM_MODEL (from .env if present)."""
    if env_file is not None:
        load_dotenv(env_file, override=False)
    model = os.environ.get("NVIDIA_VLM_MODEL") or None
    if model == "vendor/model-name-here":
        model = None
    return NvidiaSettings(
        api_key=os.environ.get("NVIDIA_API_KEY") or None,
        base_url=os.environ.get("NVIDIA_BASE_URL") or DEFAULT_BASE_URL,
        model=model,
    )


def encode_image(image: Image.Image | bytes | Path, max_side: int = MAX_IMAGE_SIDE) -> bytes:
    """Re-encode as RGB JPEG (longest side <= max_side) so payloads are small and deterministic."""
    if isinstance(image, Path | str):
        image = Path(image).read_bytes()
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    img = image.convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def cache_key(model: str, image_bytes: bytes, prompt: str) -> str:
    """SHA-256 over (model, image hash, prompt)."""
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    return hashlib.sha256(f"{model}\n{image_hash}\n{prompt}".encode()).hexdigest()


def strip_reasoning(text: str) -> str:
    """Drop <think>...</think> blocks that reasoning models prepend to the answer."""
    return re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.DOTALL).strip()


class RateLimiter:
    """Enforce a minimum interval between calls."""

    def __init__(
        self,
        requests_per_minute: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.interval = 60.0 / requests_per_minute
        self._clock, self._sleep = clock, sleep
        self._last: float | None = None

    def wait(self) -> None:
        """Block until the next call is allowed."""
        now = self._clock()
        if self._last is not None:
            delay = self._last + self.interval - now
            if delay > 0:
                self._sleep(delay)
                now += delay
        self._last = now


class ResponseCache:
    """One JSON file per response under cache_dir."""

    def __init__(self, cache_dir: Path = CACHE_DIR) -> None:
        self.cache_dir = Path(cache_dir)

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> dict | None:
        """Cached entry or None."""
        path = self._path(key)
        return json.loads(path.read_text()) if path.exists() else None

    def put(self, key: str, entry: dict) -> None:
        """Write an entry atomically."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._path(key).with_suffix(".tmp")
        tmp.write_text(json.dumps(entry, indent=2))
        tmp.replace(self._path(key))


@dataclass
class VLMResponse:
    """Answer text plus bookkeeping."""

    text: str
    model: str
    cached: bool
    latency_s: float | None


class NvidiaVLMClient:
    """Vision chat completions with throttling, retry/backoff and a disk cache."""

    def __init__(
        self,
        settings: NvidiaSettings | None = None,
        cache_dir: Path = CACHE_DIR,
        requests_per_minute: float = DEFAULT_REQUESTS_PER_MINUTE,
        max_retries: int = 5,
        backoff_base_s: float = 2.0,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings or load_settings()
        if client is None:
            if not self.settings.has_key:
                raise RuntimeError("NVIDIA_API_KEY is not configured (see .env.example)")
            client = openai.OpenAI(
                api_key=self.settings.api_key,
                base_url=self.settings.base_url,
                max_retries=0,  # retries are handled here, with logging-free backoff
                timeout=120.0,
            )
        self._client = client
        self.cache = ResponseCache(cache_dir)
        self.limiter = RateLimiter(requests_per_minute, sleep=sleep)
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self._sleep = sleep

    def list_models(self) -> list[str]:
        """IDs from the /models endpoint."""
        self.limiter.wait()
        return sorted(m.id for m in self._client.models.list().data)

    def ask(
        self,
        image: Image.Image | bytes | Path,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 512,
        use_cache: bool = True,
    ) -> VLMResponse:
        """Send one image + prompt; return the (cached when possible) answer."""
        model = model or self.settings.model
        if not model:
            raise ValueError("No model given and NVIDIA_VLM_MODEL is not set")
        image_bytes = encode_image(image)
        key = cache_key(model, image_bytes, prompt)
        if use_cache and (hit := self.cache.get(key)) is not None:
            return VLMResponse(hit["text"], model, cached=True, latency_s=hit.get("latency_s"))

        b64 = base64.b64encode(image_bytes).decode("ascii")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }
        ]
        text, latency = self._create_with_retry(model, messages, max_tokens)
        self.cache.put(
            key,
            {
                "model": model,
                "prompt": prompt,
                "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "text": text,
                "latency_s": latency,
            },
        )
        return VLMResponse(text, model, cached=False, latency_s=latency)

    def _create_with_retry(self, model: str, messages: list, max_tokens: int) -> tuple[str, float]:
        for attempt in range(self.max_retries + 1):
            self.limiter.wait()
            start = time.perf_counter()
            try:
                response = self._client.chat.completions.create(
                    model=model, messages=messages, max_tokens=max_tokens, temperature=0.0
                )
            except RETRYABLE_ERRORS as exc:
                if attempt == self.max_retries:
                    raise
                self._sleep(self._backoff(attempt, exc))
                continue
            latency = time.perf_counter() - start
            return (response.choices[0].message.content or "").strip(), latency
        raise AssertionError("unreachable")

    def _backoff(self, attempt: int, exc: Exception) -> float:
        """Retry-After header when the server sends one, else exponential backoff."""
        response = getattr(exc, "response", None)
        retry_after = response.headers.get("retry-after") if response is not None else None
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        return self.backoff_base_s * 2**attempt
