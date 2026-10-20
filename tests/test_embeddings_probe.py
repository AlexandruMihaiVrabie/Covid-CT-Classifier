"""Tests for embedding extraction/caching and the probe/zero-shot/comparison logic.

No network and no model download: a tiny fake encoder stands in for BiomedCLIP.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from ctclf import config
from ctclf import embeddings as E
from ctclf import probe as P
from ctclf.benchmark_vlm import score
from ctclf.data import Sample


class FakeModel(nn.Module):
    """Deterministic stand-in for an open_clip model; counts encoded images."""

    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.proj = nn.Linear(3, dim)
        self.images_seen = 0

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        self.images_seen += len(x)
        return self.proj(x.mean(dim=(2, 3)))

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(tokens.float())


def fake_preprocess(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.resize((8, 8)), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def fake_tokenizer(prompts: list[str]) -> torch.Tensor:
    return torch.tensor([[len(p), p.count(" "), p.count("e")] for p in prompts])


def fake_encoder() -> E.Encoder:
    return E.Encoder("fake", FakeModel().eval(), fake_preprocess, fake_tokenizer)


@pytest.fixture
def samples(tmp_path: Path) -> list[Sample]:
    out = []
    for i in range(7):
        path = tmp_path / "img" / f"x{i}.png"
        path.parent.mkdir(exist_ok=True)
        Image.new("RGB", (16, 16), (i * 30, 255 - i * 30, 50)).save(path)
        out.append(Sample(path, i % 3))
    return out


# --- embeddings ---------------------------------------------------------------------------


def test_encode_samples_shape_order_and_determinism(samples: list[Sample]) -> None:
    enc = fake_encoder()
    a = E.encode_samples(enc, samples, batch_size=3)
    b = E.encode_samples(enc, samples, batch_size=7)
    assert a.shape == (7, 4) and a.dtype == np.float32
    np.testing.assert_allclose(a, b, rtol=1e-6)
    single = E.encode_samples(enc, samples[4:5])
    np.testing.assert_allclose(a[4], single[0], rtol=1e-6)


def test_extract_split_is_resumable(samples: list[Sample], tmp_path: Path) -> None:
    out = tmp_path / "emb"
    reference = E.encode_samples(fake_encoder(), samples)

    # Simulate an interrupted run: only the first chunk is on disk.
    parts = out / "fake_parts"
    parts.mkdir(parents=True)
    np.save(parts / "test_00000.npy", reference[:3])

    enc = fake_encoder()
    matrix, computed = E.extract_split(enc, samples, "test", out, chunk_size=3, log=lambda _: None)
    assert computed == 4 and enc.model.images_seen == 4
    np.testing.assert_allclose(matrix, reference, rtol=1e-6)
    assert E.split_file(out, "fake", "test").exists()
    assert not list(parts.glob("*.npy"))  # parts merged and removed

    enc2 = fake_encoder()
    again, computed2 = E.extract_split(enc2, samples, "test", out, chunk_size=3)
    assert computed2 == 0 and enc2.model.images_seen == 0
    np.testing.assert_array_equal(again, matrix)


def test_index_roundtrip_and_alignment(samples: list[Sample], tmp_path: Path) -> None:
    splits = {"train": samples[:4], "val": samples[4:5], "test": samples[5:]}
    rows = E.index_rows(splits)
    assert [r["split"] for r in rows] == ["train"] * 4 + ["val"] + ["test"] * 2
    assert rows[1] == {"filename": "x1.png", "split": "train", "label": "pCT", "label_index": 1}

    E.write_index(tmp_path / "fake_index.csv", rows)
    assert E.read_index(tmp_path / "fake_index.csv") == rows
    np.save(E.split_file(tmp_path, "fake", "test"), np.zeros((2, 4), dtype=np.float32))

    emb, got = E.load_split_embeddings("test", "fake", tmp_path, expected=splits["test"])
    assert emb.shape == (2, 4) and [r["filename"] for r in got] == ["x5.png", "x6.png"]
    with pytest.raises(ValueError, match="does not match"):
        E.load_split_embeddings("test", "fake", tmp_path, expected=samples[:2])
    np.save(E.split_file(tmp_path, "fake", "test"), np.zeros((3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="index rows"):
        E.load_split_embeddings("test", "fake", tmp_path)


def test_text_embeddings_cached(tmp_path: Path) -> None:
    calls = []

    def factory() -> E.Encoder:
        calls.append(1)
        return fake_encoder()

    first = E.cached_text_embeddings(["a b", "c d e"], "fake", tmp_path, encoder_factory=factory)
    assert first.shape == (2, 4) and len(calls) == 1

    def boom() -> E.Encoder:
        raise AssertionError("encoder must not load on a cache hit")

    second = E.cached_text_embeddings(["c d e", "a b"], "fake", tmp_path, encoder_factory=boom)
    np.testing.assert_allclose(second, first[::-1], rtol=1e-6)
    E.cached_text_embeddings(["new one"], "fake", tmp_path, encoder_factory=factory)
    assert len(calls) == 2


def test_timing_test_report(samples: list[Sample]) -> None:
    result = E.timing_test(fake_encoder(), samples, 4, "cpu", 2, 0, {"all": 1000})
    assert result["n_images"] == 4
    assert result["seconds_per_image"] > 0
    assert result["estimated_minutes"]["all"] == pytest.approx(
        1000 * result["seconds_per_image"] / 60
    )


# --- probe and zero-shot ------------------------------------------------------------------


def synthetic(n_per_class: int, seed: int) -> tuple[np.ndarray, list[int]]:
    rng = np.random.default_rng(seed)
    centres = np.eye(3, 8) * 3
    x = np.concatenate([centres[k] + rng.normal(0, 0.5, (n_per_class, 8)) for k in range(3)])
    return x.astype(np.float32), [k for k in range(3) for _ in range(n_per_class)]


def test_l2_normalize() -> None:
    x = P.l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    np.testing.assert_allclose(x, [[0.6, 0.8], [0.0, 0.0]])


def test_probe_selection_and_determinism() -> None:
    x_tr, y_tr = synthetic(40, 0)
    x_va, y_va = synthetic(20, 1)
    params, grid = P.select_probe(x_tr, y_tr, x_va, y_va, c_grid=(0.1, 1.0), class_weights=(None,))
    assert len(grid) == 2 and params["C"] in (0.1, 1.0)
    clf = P.fit_probe(x_tr, y_tr, params["C"], params["class_weight"])
    preds = P.predict_probe(clf, x_va)
    assert score(y_va, preds)["accuracy"] > 0.95
    again = P.predict_probe(P.fit_probe(x_tr, y_tr, params["C"], None), x_va)
    assert preds == again


def test_select_probe_tie_breaks_to_smaller_c() -> None:
    x_tr, y_tr = synthetic(30, 2)
    params, _ = P.select_probe(x_tr, y_tr, x_tr, y_tr, c_grid=(10.0, 100.0), class_weights=(None,))
    assert params["C"] == 10.0  # both are perfect on this easy data


def test_zero_shot_argmax_and_prompt_averaging() -> None:
    text = np.array([[1, 0, 0], [1, 0.2, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    class_emb = P.class_text_embeddings(text, [0, 0, 1, 2])
    np.testing.assert_allclose(np.linalg.norm(class_emb, axis=1), 1.0, rtol=1e-6)
    assert class_emb[0, 1] > 0  # average of the two nCT prompts
    images = np.array([[5, 0.1, 0], [0, 2, 0.5], [0.1, 0, 9], [0, 0.4, 0.3]])
    assert P.zero_shot_predict(images, class_emb) == [0, 1, 2, 1]


def test_flatten_prompts_class_order() -> None:
    prompts, labels = P.flatten_prompts(P.PROMPT_SETS["descriptive"])
    assert len(prompts) == 6 and labels == [0, 0, 1, 1, 2, 2]
    for prompt_set in P.PROMPT_SETS.values():
        assert set(prompt_set) == set(config.CLASSES)


@pytest.mark.parametrize(
    ("b", "c", "p"),
    [
        (0, 0, 1.0),
        (5, 0, 2 / 32),  # 2 * C(5,0) / 2^5
        (1, 9, 22 / 1024),  # 2 * (C(10,0) + C(10,1)) / 2^10
        (3, 3, 1.0),  # capped at 1
        (2, 6, 2 * (1 + 8 + 28) / 256),
    ],
)
def test_mcnemar_exact_hand_checked(b: int, c: int, p: float) -> None:
    ref = [True] * b + [False] * c + [True] * 4 + [False] * 2
    other = [False] * b + [True] * c + [True] * 4 + [False] * 2
    out = P.mcnemar_exact(ref, other)
    assert out["ref_only_correct"] == b and out["other_only_correct"] == c
    assert out["n_pairs"] == b + c + 6
    assert out["p_value"] == pytest.approx(p)


def test_with_cis_from_confusion_matrix() -> None:
    m = score([0, 0, 1, 2, 2], [0, 1, 1, 2, 0])
    stripped = {k: m[k] for k in ("n", "accuracy", "macro_f1", "confusion_matrix")}
    stripped["per_class"] = {
        cls: {k: v for k, v in pc.items() if k not in ("correct", "recall_ci95")}
        for cls, pc in m["per_class"].items()
    }
    rebuilt = P.with_cis(stripped)
    assert rebuilt["correct"] == m["correct"] == 3
    assert list(rebuilt["accuracy_ci95"]) == list(m["accuracy_ci95"])
    for cls in config.CLASSES:
        assert list(rebuilt["per_class"][cls]["recall_ci95"]) == list(
            m["per_class"][cls]["recall_ci95"]
        )


def test_duplicated_in_train(tmp_path: Path) -> None:
    def img(name: str, colour: int) -> Sample:
        path = tmp_path / name
        Image.new("RGB", (4, 4), (colour, 0, 0)).save(path)
        return Sample(path, 0)

    train = [img("a.png", 1), img("b.png", 2)]
    test = [img("c.png", 2), img("d.png", 3)]
    assert P.duplicated_in_train(train, test) == ["c.png"]


def test_cnn_recheck() -> None:
    saved = score([0, 1, 2], [0, 1, 1])
    assert P.cnn_recheck(score([0, 1, 2], [0, 1, 1]), saved)["passed"]
    assert not P.cnn_recheck(score([0, 1, 2], [0, 1, 2]), saved)["passed"]


def test_comparison_consistency(tmp_path: Path) -> None:
    y = [0, 0, 1, 1, 2, 2]
    m_cnn, m_probe, m_zs, m_vlm = (
        score(y, p)
        for p in ([0, 0, 1, 1, 2, 2], [0, 0, 1, 2, 2, 2], [0, 1, 1, 0, 2, 2], [0, 0, -1, 0, 2, 2])
    )
    probe_summary = {
        "encoder": "fake",
        "probe": {"test": m_probe},
        "zero_shot": {"test": m_zs},
        "mcnemar_cnn_vs_probe_full_test": P.mcnemar_exact([True] * 6, [True] * 5 + [False]),
        "subset_100": {"probe": m_probe, "zero_shot": m_zs},
        "duplicate_sensitivity": {
            "test_images_with_exact_duplicate_in_train": 1,
            "n_without": 5,
            "cnn_without": m_cnn,
            "probe_without": m_probe,
        },
    }
    test_aggregate = {k: m_cnn[k] for k in ("n", "accuracy", "macro_f1", "confusion_matrix")}
    test_aggregate["per_class"] = {c: {"recall": 1.0} for c in config.CLASSES}
    metrics = {"splits": {"test": test_aggregate}}
    vlm_summary = {
        "n": 6,
        "per_class_n": {c: 2 for c in config.CLASSES},
        "vlm_model": "x/y",
        "vlm_unparseable": 1,
        "cnn": m_cnn,
        "vlm": m_vlm,
    }
    comp = P.build_comparison(probe_summary, metrics, vlm_summary)
    assert list(comp["subset_100"]["models"]) == [
        "cnn",
        "fake_probe",
        "fake_zero_shot",
        "nvidia_vlm_zero_shot",
    ]
    assert comp["full_test"]["models"]["cnn"]["accuracy_ci95"] == m_cnn["accuracy_ci95"]
    assert comp["subset_100"]["models"]["nvidia_vlm_zero_shot"]["accuracy"] == pytest.approx(4 / 6)

    md = P.comparison_markdown(comp)
    assert "| fake_probe |" in md and "McNemar" in md and "byte-identical" in md

    for name, obj in [
        (P.PROBE_SUMMARY_NAME, probe_summary),
        ("metrics.json", metrics),
        ("vlm_benchmark_summary.json", vlm_summary),
        (P.COMPARISON_NAME, comp),
    ]:
        (tmp_path / name).write_text(json.dumps(obj))
    assert P.check_files(tmp_path)
    comp["full_test"]["n"] = 999
    (tmp_path / P.COMPARISON_NAME).write_text(json.dumps(comp))
    assert not P.check_files(tmp_path)
