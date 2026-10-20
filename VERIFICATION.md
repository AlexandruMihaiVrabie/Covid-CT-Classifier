# Verification report

Stage 8 of the project plan: every check below was actually run on 2026-10-20, and the
output shown is copied from the terminal. Machine: MacBook Air M1, 8 GB RAM, macOS
(Darwin 24.6.0), Python 3.11.0. Code state: commit `4af9396` ("feat: gradio app") plus this stage's changes
(`app/smoke_check.py`, one extra test, this file).

**Summary: all checks passed. No failures had to be fixed during this stage.**

| Check | Result |
|---|---|
| Fresh venv install from `requirements.txt` | OK in 7 min 25 s, `pip check` clean |
| `ruff format --check` / `ruff check` | clean |
| `pytest` (fresh venv) | 79 passed (the 80th test was added afterwards; main venv: 80 passed) |
| Coverage, all code (`src/ctclf` + `app`) | 73% |
| Coverage, core logic (CLI `main()` excluded) | 95% |
| Re-run of data check, evaluation and probe | results identical to the committed files |
| End-to-end: CLI predict, Grad-CAM, probe, app | all OK |

## 1. Fresh install

Following the setup instructions (the same ones that go into the README), in a new, empty venv:

```bash
python3.11 -m venv freshvenv && source freshvenv/bin/activate
pip install -r requirements.txt && pip install -e .
```

```
( pip install -q -r requirements.txt && pip install -q -e .; )  ... 7:24.95 total
exit 0
$ pip check
No broken requirements found.
$ python -c "import ctclf, torch, open_clip, gradio; ..."
imports ok 2.14.1 3.3.0 6.30.0
$ pytest
79 passed in 78.43s (0:01:18)
```

## 2. Lint and format

```
$ ruff format --check .
28 files already formatted
$ ruff check .
All checks passed!
```

## 3. Tests and coverage

```
$ pytest --cov=src/ctclf --cov=app
Name                         Stmts   Miss  Cover
------------------------------------------------
app/app.py                      60      8    87%
app/smoke_check.py              37     19    49%
src/ctclf/__init__.py            1      0   100%
src/ctclf/benchmark_vlm.py     121     39    68%
src/ctclf/config.py             27      4    85%
src/ctclf/data.py              118     22    81%
src/ctclf/discover_vlm.py       48     31    35%
src/ctclf/embeddings.py        156     42    73%
src/ctclf/evaluate.py          128     37    71%
src/ctclf/gradcam.py            94     28    70%
src/ctclf/model.py              19      0   100%
src/ctclf/nvidia_client.py     136      7    95%
src/ctclf/predict.py            42      5    88%
src/ctclf/probe.py             191     84    56%
src/ctclf/train.py             111     23    79%
------------------------------------------------
TOTAL                         1289    349    73%
80 passed in 52.04s
```

Almost all uncovered lines are the command-line `main()` functions. They need the real
dataset, the MPS GPU, the downloaded BiomedCLIP weights or the NVIDIA API, so unit tests
(which use no dataset, no network and no download) do not run them. They are exercised
by the end-to-end runs in sections 4 and 5 instead. Coverage of the core logic, with
`main()` and `__main__` blocks excluded, is **95%** (`app/app.py` 100%, `probe.py` 96%,
`embeddings.py` 95%):

```bash
printf '[run]\nsource = src/ctclf,app\n[report]\nexclude_lines =\n    pragma: no cover\n    def main\\(\n    if __name__ == .__main__.:\n' > core.coveragerc
pytest --cov --cov-config=core.coveragerc     # TOTAL 952 stmts, 45 missed, 95%
```

## 4. Reproducibility of the committed results

| Command | Check | Result |
|---|---|---|
| `python -m ctclf.data` | `git diff outputs/results/data_report.json` | 0 lines changed |
| `python -m ctclf.evaluate --results-dir <tmp>` | new `metrics.json` vs committed (except the checkpoint path) | identical |
| `python -m ctclf.probe` (from cached embeddings) | `git diff outputs/` | no change: `probe_summary.json`, `comparison.json`, `comparison.md`, `test_predictions.csv` byte-identical |
| `python -m ctclf.benchmark_vlm --check` | VLM summary recomputed from raw rows | `Summary matches raw file: PASS` |
| `python -m ctclf.probe --check` | `comparison.json` rebuilt from its source files | `PASS` |

Checks built into the probe run:

```
Selected probe {'C': 10.0, 'class_weight': None} (val macro-F1 0.9684)
Selected prompt set 'biomedclip_template'
CNN recomputed vs metrics.json: PASS
CNN on the 100 slices matches vlm_benchmark_raw.json: True
comparison.json matches its source files: PASS
```

Evaluation re-run (best.pth, epoch 10):

```
train n=13778 loss 0.0097 acc 0.9961 macro-F1 0.9961
val   n= 2951 loss 0.0221 acc 0.9936 macro-F1 0.9935
test  n= 2956 loss 0.0182 acc 0.9922 macro-F1 0.9916
Test confusion matrix (rows true, cols pred ('nCT', 'pCT', 'NiCT')): [[1488, 2, 8], [0, 597, 4], [6, 3, 848]]
Majority baseline (always nCT): acc 0.5068 macro-F1 0.2242
Consistency check vs training log (epoch 10): PASS abs_diff={'loss': 0.0, 'accuracy': 0.0, 'macro_f1': 0.0}
```

## 5. End-to-end sample run

Test-split slice `pCT310.jpg` (not one of the app examples):

```
$ python -m ctclf.predict "archive/Original CT Scans/pCT/pCT310.jpg"
pCT310.jpg: pCT (COVID-19 positive)
  nCT   0.0000
  pCT   0.9997
  NiCT  0.0002
Educational use only, not a medical device.

$ python -m ctclf.gradcam "archive/Original CT Scans/pCT/pCT310.jpg" --out pCT310_gradcam.png
pCT310.jpg: pCT (COVID-19 positive) -> pCT310_gradcam.png      # 224x224 RGB overlay
```

In `test_predictions.csv` the same slice is `pCT310.jpg,pCT,pCT,pCT,nCT`
(true, CNN, probe, zero-shot): the CNN and the probe are right, BiomedCLIP zero-shot is wrong.

**App, real call.** The app was started from the fresh venv with an empty environment
(`env -i`, so no `NVIDIA_*` variables and no `.env` loading), on port 7861, then called
through `gradio_client`:

```bash
env -i HOME="$HOME" PATH=/usr/bin:/bin freshvenv/bin/python app/app.py --port 7861 &
python app/smoke_check.py --url http://127.0.0.1:7861/
```

```
* Running on local URL:  http://127.0.0.1:7861
$ lsof -iTCP:7861 -sTCP:LISTEN
Python  96131 ...  TCP 127.0.0.1:7861 (LISTEN)
nCT7356.jpg        -> nCT  [nCT 1.0000, NiCT 0.0000, pCT 0.0000] overlay (224, 224, 3) OK
pCT2369.jpg        -> pCT  [pCT 0.9991, nCT 0.0005, NiCT 0.0005] overlay (224, 224, 3) OK
NiCT219.jpg        -> NiCT [NiCT 1.0000, nCT 0.0000, pCT 0.0000] overlay (224, 224, 3) OK
non_ct_colour.png  -> nCT  [nCT 0.9174, NiCT 0.0774, pCT 0.0052] overlay (224, 224, 3) OK
non_ct_gray.png    -> NiCT [NiCT 0.6979, pCT 0.2467, nCT 0.0554] overlay (224, 224, 3) OK
ALL OK
```

The server listened on 127.0.0.1 only. A unit test also checks that loading the app
imports neither `dotenv`, `openai` nor `ctclf.nvidia_client`. The two non-CT images show
the expected limitation: a synthetic colour picture is labelled "normal" with 91.7%
probability, because the model can only answer with one of its three classes. The UI
states this next to the disclaimer.

## 6. Verified results (from `outputs/results/comparison.json`)

Full test split (n = 2,956), Wilson 95% confidence intervals:

| Model | Accuracy [95% CI] | Macro-F1 |
|---|---|---|
| CNN (MobileNetV3-Large) | 99.2% [98.8%, 99.5%] (2933/2956) | 0.992 |
| BiomedCLIP + logistic regression | 97.5% [96.9%, 98.0%] (2882/2956) | 0.972 |
| BiomedCLIP zero-shot | 85.9% [84.6%, 87.1%] (2538/2956) | 0.854 |

Exact McNemar, CNN vs probe: 65 slices only the CNN got right, 14 only the probe,
p = 5.19e-09. On the 100 slices of the NVIDIA benchmark: CNN 100%, probe 95%,
BiomedCLIP zero-shot 80%, NVIDIA VLM zero-shot (`nvidia/nemotron-3.5-super-vl-120b-a12b`)
65% [55.3%, 73.6%]. Removing the 15 test images that have a byte-identical copy in train
leaves the CNN at 99.22% and the probe at 97.48% (n = 2,941).

## 7. What could NOT be verified

- **Clinical validity.** Nothing here shows the models are safe or useful for diagnosis.
  This is an educational project, not a medical device.
- **Generalisation to other hospitals, scanners or protocols.** All data come from one
  public dataset (two hospitals in Wuhan). No external test set was available.
- **Optimism of the per-slice split.** The dataset has no patient identifiers, so slices of
  the same patient can be in train and test. The test accuracy is therefore optimistic,
  and how optimistic cannot be measured here. Only exact byte duplicates could be checked
  (section 6).
- **BiomedCLIP on other CT sources.** The probe and zero-shot numbers hold for this dataset
  and this split only.
- **Availability of the NVIDIA model over time.** The VLM benchmark was not re-run in this
  stage (by design: no new API calls). Only the consistency of its saved summary with its
  raw responses was checked. Hosted free models can change or be withdrawn, so the 65% may
  not be reproducible later.
- **Training reproducibility.** Training was not re-run. The committed `best.pth` was
  re-evaluated and matches its training log exactly, but a new training run on MPS may not
  give identical weights.
- **Embedding extraction was not re-run on the full dataset** (about 19 minutes). The
  cached embeddings were reused. Earlier checks showed MPS and CPU embeddings agree to
  within 5e-5. Resuming an interrupted extraction is covered by unit tests only.
- **The app was not tested with the network physically disabled.** Verified instead: no
  environment variables, no API or `.env` modules imported, CNN weights loaded from the
  repository, server bound to 127.0.0.1.
- **README numbers.** The README is written in the next stage; its numbers will be checked
  against `comparison.json` and `metrics.json` there.
