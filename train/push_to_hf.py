#!/usr/bin/env python3
"""Push all trained models to their HuggingFace repos.

Repos:
  TigreGotico/eat-classifiers       — EAT ONNX models + benchmarks
  TigreGotico/sentence-types        — sentence-type ONNX models + benchmarks
  TigreGotico/yes-no-classifiers    — yes/no ONNX models + benchmarks

Usage::

    python -m train.push_to_hf                   # push all three repos
    python -m train.push_to_hf --repo eat        # eat-classifiers only
    python -m train.push_to_hf --repo sentence   # sentence-types only
    python -m train.push_to_hf --repo yesno      # yes-no-classifiers only
    python -m train.push_to_hf --dry-run         # print what would be uploaded
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from os.path import dirname
from pathlib import Path
from textwrap import dedent

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_TRAIN_DIR = Path(dirname(__file__))
_ROOT_DIR = _TRAIN_DIR.parent
_BUNDLED = _ROOT_DIR / "little_questions" / "models"

_CACHE = Path(os.path.expanduser("~/.local/share/little_questions"))

EAT_ONNX_DIR = _CACHE / "eat"
YESNO_ONNX_DIR = _CACHE / "yesno"
SENTENCE_TYPE_ONNX_DIR = _CACHE / "sentence_type"

EAT_REPORTS_DIR = _TRAIN_DIR / "reports" / "eat"
SENTENCE_TYPE_REPORTS_DIR = _TRAIN_DIR / "reports" / "sentence_type"
YESNO_REPORTS_DIR = _TRAIN_DIR / "reports" / "yesno"

VERSION = "0.9.0"

HF_EAT_REPO = "TigreGotico/eat-classifiers"
HF_SENTENCE_REPO = "TigreGotico/sentence-types"
HF_YESNO_REPO = "TigreGotico/yes-no-classifiers"


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------

def _upload_file(api, local_path: Path, repo_id: str, repo_path: str, dry_run: bool) -> None:
    print(f"  {'[dry] ' if dry_run else ''}upload {repo_id}:{repo_path}")
    if not dry_run:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="model",
        )


def _upload_text(api, content: str, repo_id: str, repo_path: str, dry_run: bool) -> None:
    print(f"  {'[dry] ' if dry_run else ''}upload {repo_id}:{repo_path}")
    if not dry_run:
        api.upload_file(
            path_or_fileobj=content.encode(),
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="model",
        )


# ---------------------------------------------------------------------------
# EAT repo
# ---------------------------------------------------------------------------

def _eat_readme(onnx_files: list[Path]) -> str:
    rows = []
    for f in sorted(onnx_files):
        stem = f.stem
        punctuated = "unpunct" not in stem
        variant = "punctuated (written)" if punctuated else "unpunctuated (ASR/voice)"
        calibrated = "cal" in stem
        output = "calibrated probability" if calibrated else "decision score"
        rows.append(f"| `{f.name}` | {variant} | {output} |")
    table = "\n".join(rows) or "_No models found_"

    tmpl = """\
---
language:
  - en
license: apache-2.0
tags:
  - question-classification
  - text-classification
  - onnx
  - english
  - eat
  - calibrated
datasets:
  - TigreGotico/EAT
---

# eat-classifiers

English question answer-type (EAT) classifiers trained on the
[TigreGotico/EAT](https://huggingface.co/datasets/TigreGotico/EAT) dataset
(30,017 questions, 53 fine-grained labels across 7 TREC categories).

Two-stage inference (eat7 gates eat53) achieves **93.4% macro F1** on the test set.

Used by [little_questions](https://github.com/OpenJarbas/little_questions).

## Label taxonomy

7 main categories, 53 sub-types:

| Main | Sub-types |
|------|-----------|
| `ABBR` | abb, exp |
| `BOOL` | yesno |
| `DESC` | def, desc, manner, reason |
| `ENTY` | animal, body, color, cremat, currency, dismed, event, food, instru, lang, letter, other, plant, product, religion, sport, substance, symbol, techmeth, termeq, veh, word |
| `HUM` | desc, gr, ind, title |
| `LOC` | city, country, landmass, mount, other, state, water |
| `NUM` | code, count, date, dist, money, ord, other, perc, period, speed, temp, volsize, weight |

## Models

| File | Input variant | Output[1] |
|------|---------------|-----------|
TABLE_ROWS

Both punctuated and unpunctuated variants are provided.
Use the unpunctuated (`_unpunct`) model for ASR / voice assistant input.

## Two-stage inference

```python
import onnxruntime as rt, numpy as np, json

sess7  = rt.InferenceSession("eat7_svm_cal_EN_VERSION.onnx")
sess53 = rt.InferenceSession("eat53_svm_cal_EN_VERSION.onnx")
classes7  = json.loads(sess7.get_modelmeta().custom_metadata_map["classes"])
classes53 = json.loads(sess53.get_modelmeta().custom_metadata_map["classes"])
main_of_53 = [c.split(":")[0] for c in classes53]

def classify(text):
    inp = np.array([text], dtype=object)
    main = classes7[int(sess7.run(None, {"input": inp})[0][0])]
    _, probs = sess53.run(None, {"input": inp})
    row = probs[0].copy()
    for j, m in enumerate(main_of_53):
        if m != main:
            row[j] = 0.0
    row /= row.sum()
    return classes53[int(np.argmax(row))], float(row.max())

print(classify("Who invented the telephone?"))  # ('HUM:ind', 0.96)
```

## Benchmarks

Full results: [BENCHMARKS.md](BENCHMARKS.md)
"""
    return tmpl.replace("TABLE_ROWS", table).replace("VERSION", VERSION)


def _eat_benchmarks() -> str:
    rows = []
    if EAT_REPORTS_DIR.exists():
        for rf in sorted(EAT_REPORTS_DIR.glob("*_benchmark.json")):
            try:
                d = json.loads(rf.read_text())
                name = d.get("scorer", rf.stem.replace("_benchmark", ""))
                acc = d.get("accuracy", 0)
                mf1 = d.get("macro_f1", 0)
                rows.append((mf1, f"| `{name}` | {acc:.4f} | {mf1:.4f} |"))
            except Exception:
                pass
    header = "| Model | Accuracy | Macro F1 |\n|-------|----------|----------|"
    body = "\n".join(r for _, r in sorted(rows, reverse=True))
    return (
        "# EAT Classifier Benchmarks\n\n"
        "Test set: 15% stratified split of TigreGotico/EAT (4,503 samples, random_state=42).\n\n"
        + header + "\n" + body + "\n\n"
        "## Plots\n\n"
        "![Overview](benchmarks/benchmark_eat53_overview.png)\n"
        "![Model comparison](benchmarks/benchmark_eat_model_comparison.png)\n"
        "![Per-class F1 53-class](benchmarks/benchmark_eat53_per_class_f1.png)\n"
        "![Confusion 7-class](benchmarks/benchmark_eat7_confusion.png)\n"
    )


def push_eat(api, dry_run: bool) -> None:
    onnx_files = sorted(EAT_ONNX_DIR.glob("*.onnx")) if EAT_ONNX_DIR.exists() else []
    # also include bundled files not yet in cache
    for bf in (_BUNDLED / "eat").glob("*.onnx"):
        if not any(f.name == bf.name for f in onnx_files):
            onnx_files.append(bf)

    if not dry_run:
        api.create_repo(repo_id=HF_EAT_REPO, repo_type="model", exist_ok=True)

    _upload_text(api, _eat_readme(onnx_files), HF_EAT_REPO, "README.md", dry_run)
    _upload_text(api, _eat_benchmarks(), HF_EAT_REPO, "BENCHMARKS.md", dry_run)

    print(f"\n  EAT ONNX models ({len(onnx_files)} files):")
    for f in sorted(onnx_files, key=lambda x: x.name):
        _upload_file(api, f, HF_EAT_REPO, f"models/eat/{f.name}", dry_run)

    if EAT_REPORTS_DIR.exists():
        plots = list(EAT_REPORTS_DIR.glob("*.png"))
        jsons = list(EAT_REPORTS_DIR.glob("*.json"))
        print(f"\n  EAT benchmarks ({len(plots)} plots, {len(jsons)} JSON):")
        for f in sorted(plots + jsons):
            _upload_file(api, f, HF_EAT_REPO, f"benchmarks/{f.name}", dry_run)


# ---------------------------------------------------------------------------
# Sentence-type repo
# ---------------------------------------------------------------------------

def _sentence_readme() -> str:
    langs = {
        "EN": "English", "DE": "German", "ES": "Spanish", "FR": "French",
        "IT": "Italian", "NL": "Dutch", "PT": "Portuguese",
    }
    rows = "\n".join(
        f"| `sentence_type_{code}_0.8.0.onnx` | {name} |"
        for code, name in langs.items()
    )
    tmpl = """\
---
language:
  - en
  - de
  - es
  - fr
  - it
  - nl
  - pt
license: apache-2.0
tags:
  - sentence-classification
  - text-classification
  - onnx
  - multilingual
datasets:
  - TigreGotico/sentence-types-multilingual
---

# sentence-types

Multilingual sentence-type classifiers (ONNX) trained on
[TigreGotico/sentence-types-multilingual](https://huggingface.co/datasets/TigreGotico/sentence-types-multilingual)
(9,900 balanced samples per language, 6 classes).

Used by [little_questions](https://github.com/OpenJarbas/little_questions).

## Classes

`command`, `exclamation`, `polar_question`, `request`, `statement`, `wh_question`

## Models

| File | Language |
|------|----------|
MODEL_ROWS

## Accuracy

| Language | Accuracy | Macro F1 |
|----------|----------|----------|
| EN | 99.2% | 99.2% |
| NL | 98.8% | 98.8% |
| FR | 97.1% | 97.1% |
| IT | 97.0% | 97.0% |
| PT | 95.4% | 95.4% |
| DE | 85.6% | 84.9% |
| ES | 74.6% | 72.7% |

## Inference

```python
import onnxruntime as rt, numpy as np, json

sess = rt.InferenceSession("sentence_type_EN_0.8.0.onnx")
classes = json.loads(sess.get_modelmeta().custom_metadata_map["classes"])
inp = np.array(["Who invented the telephone?"], dtype=object)
label_idx, probs = sess.run(None, {"input": inp})
print(classes[int(label_idx[0])])   # wh_question
```
"""
    return tmpl.replace("MODEL_ROWS", rows)


def push_sentence_type(api, dry_run: bool) -> None:
    onnx_files: list[Path] = []
    for d in (SENTENCE_TYPE_ONNX_DIR, _BUNDLED / "sentence_type"):
        if d.exists():
            for f in d.glob("*.onnx"):
                if not any(x.name == f.name for x in onnx_files):
                    onnx_files.append(f)

    if not dry_run:
        api.create_repo(repo_id=HF_SENTENCE_REPO, repo_type="model", exist_ok=True)

    _upload_text(api, _sentence_readme(), HF_SENTENCE_REPO, "README.md", dry_run)

    print(f"\n  Sentence-type ONNX models ({len(onnx_files)} files):")
    for f in sorted(onnx_files, key=lambda x: x.name):
        _upload_file(api, f, HF_SENTENCE_REPO, f"models/{f.name}", dry_run)

    if SENTENCE_TYPE_REPORTS_DIR.exists():
        plots = list(SENTENCE_TYPE_REPORTS_DIR.glob("*.png"))
        jsons = list(SENTENCE_TYPE_REPORTS_DIR.glob("*.json"))
        print(f"\n  Sentence-type benchmarks ({len(plots)} plots, {len(jsons)} JSON):")
        for f in sorted(plots + jsons):
            _upload_file(api, f, HF_SENTENCE_REPO, f"benchmarks/{f.name}", dry_run)


# ---------------------------------------------------------------------------
# Yes/No repo
# ---------------------------------------------------------------------------

def _yesno_readme(onnx_files: list[Path]) -> str:
    per_lang = [fn for fn in sorted(onnx_files, key=lambda fn: fn.name) if "multilingual" not in fn.name]
    lang_rows = "\n".join("| `" + fn.name + "` |" for fn in per_lang)
    tmpl = """\
---
language:
  - multilingual
license: apache-2.0
tags:
  - text-classification
  - sentiment-analysis
  - onnx
  - multilingual
  - yes-no
datasets:
  - TigreGotico/yes-no-multilingual
---

# yes-no-classifiers

Yes/No answer polarity classifiers (ONNX) for 43 languages, trained on
[TigreGotico/yes-no-multilingual](https://huggingface.co/datasets/TigreGotico/yes-no-multilingual)
(200 samples/language, 8,600 total).

Used by [little_questions](https://github.com/OpenJarbas/little_questions) to detect
whether a statement is an affirmative, negative, or uncertain answer.

## Classes

`yes`, `no`, `maybe`

## Models

A **multilingual model** (`yesno_svm_cal_multilingual_VERSION.onnx`) covers all 43
languages and ships bundled with `little_questions`. Per-language models achieve
90-96% macro F1 on their own language; the multilingual model achieves 84%.

### Per-language models

| File |
|------|
LANG_ROWS

## Inference

```python
import onnxruntime as rt, numpy as np

sess = rt.InferenceSession("yesno_svm_cal_multilingual_VERSION.onnx")
inp = np.array(["Yes, of course!"], dtype=object)
label, probs = sess.run(None, {"input": inp})
print(label[0])   # yes
```
"""
    return tmpl.replace("VERSION", VERSION).replace("LANG_ROWS", lang_rows)


def push_yesno(api, dry_run: bool) -> None:
    onnx_files: list[Path] = []
    for d in (YESNO_ONNX_DIR, _BUNDLED / "yesno"):
        if d.exists():
            for f in d.glob("*.onnx"):
                if not any(x.name == f.name for x in onnx_files):
                    onnx_files.append(f)

    if not dry_run:
        api.create_repo(repo_id=HF_YESNO_REPO, repo_type="model", exist_ok=True)

    _upload_text(api, _yesno_readme(onnx_files), HF_YESNO_REPO, "README.md", dry_run)

    print(f"\n  Yes/No ONNX models ({len(onnx_files)} files):")
    for f in sorted(onnx_files, key=lambda x: x.name):
        _upload_file(api, f, HF_YESNO_REPO, f"models/yesno/{f.name}", dry_run)

    if YESNO_REPORTS_DIR.exists():
        plots = list(YESNO_REPORTS_DIR.glob("*.png"))
        jsons = list(YESNO_REPORTS_DIR.glob("*.json"))
        print(f"\n  Yes/No benchmarks ({len(plots)} plots, {len(jsons)} JSON):")
        for f in sorted(plots + jsons):
            _upload_file(api, f, HF_YESNO_REPO, f"benchmarks/{f.name}", dry_run)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Push models to HuggingFace")
    parser.add_argument("--repo", choices=["eat", "sentence", "yesno", "all"], default="all",
                        help="Which repo to push (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be uploaded")
    args = parser.parse_args()

    from huggingface_hub import HfApi
    api = HfApi() if not args.dry_run else None

    do_eat = args.repo in ("eat", "all")
    do_sentence = args.repo in ("sentence", "all")
    do_yesno = args.repo in ("yesno", "all")

    if do_eat:
        print(f"\n{'='*60}\nPushing {HF_EAT_REPO}\n{'='*60}")
        push_eat(api, args.dry_run)

    if do_sentence:
        print(f"\n{'='*60}\nPushing {HF_SENTENCE_REPO}\n{'='*60}")
        push_sentence_type(api, args.dry_run)

    if do_yesno:
        print(f"\n{'='*60}\nPushing {HF_YESNO_REPO}\n{'='*60}")
        push_yesno(api, args.dry_run)

    if not args.dry_run:
        print("\nDone.")
        if do_eat:
            print(f"  https://huggingface.co/{HF_EAT_REPO}")
        if do_sentence:
            print(f"  https://huggingface.co/{HF_SENTENCE_REPO}")
        if do_yesno:
            print(f"  https://huggingface.co/{HF_YESNO_REPO}")


if __name__ == "__main__":
    main()
