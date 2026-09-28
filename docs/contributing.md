# Contributing & Development Guide

## Setup

```bash
git clone https://github.com/TigreGotico/little_questions
cd little_questions
pip install -e ".[train]"
```

## Running tests

```bash
# Unit tests (no models required)
pytest test/
# Integration tests (require downloaded ONNX models)
pytest test/ --integration
```

## Project layout

```
little_questions/
├── __init__.py      # Sentence, subclasses, get_classifier(), get_scorer(), get_yesno_classifier()
├── classifiers.py   # EatClassifier, SentenceTypeClassifier, YesNoClassifier, _OnnxModel
├── constants.py     # EAT_LABELS_7, EAT_LABELS_53, SENTENCE_TYPES, MAIN_LABEL_NAMES, SEC_LABEL_NAMES
└── models.py        # HF auto-download helpers
train/               # Training only. Install with pip install little-questions[train]
├── classifiers.py       # CalibratedLinearSVCClassifier, LinearSVCClassifier, LogRegClassifier,
│                        # SGDClassifier, Model2VecClassifier, EATTextPreprocessor
├── load_eat.py          # EAT dataset loader (HF + local TSV)
├── load_yesno.py        # Yes/no dataset loader (HF)
├── train_eat.py         # Train EAT ONNX baselines (svm_cal + uncalibrated variants)
├── train_eat_m2v.py     # Train Model2Vec EAT variants (12 models)
├── train_yesno.py       # Train yes/no polarity classifiers → ONNX
├── train_sentence_type.py
├── benchmark_eat.py     # Full EAT benchmark + 6 plots
├── metrics.py           # EvalResult, evaluate(), compare()
├── mlflow_config.py     # MLflow setup
└── push_to_hf.py        # Push models + benchmarks to HuggingFace
```

## Training

```bash
# EAT classifiers (calibrated ONNX, both punctuated + ASR variants)
python -m train.train_eat
# EAT Model2Vec variants
python -m train.train_eat_m2v
# Yes/no polarity classifiers (per-language + multilingual)
python -m train.train_yesno
# Sentence-type classifiers
python -m train.train_sentence_type
# Benchmarks + plots
python -m train.benchmark_eat
python -m train.train_yesno --plot
# Push all models to HuggingFace
python -m train.push_to_hf
```

## HuggingFace repos

Model repos:

| Repo | Contents |
|------|----------|
| `TigreGotico/eat-classifiers` | EAT ONNX models + benchmarks |
| `TigreGotico/sentence-types` | Sentence-type ONNX models |
| `TigreGotico/yes-no-classifiers` | Yes/no ONNX models |

Training data repos:

| Repo | Contents |
|------|----------|
| `TigreGotico/EAT` | EAT training dataset (30K EN, 53 labels) |
| `TigreGotico/sentence-types-multilingual` | Sentence-type training data (80K) |
| `TigreGotico/yes-no-multilingual` | Yes/no training data (8.6K, 43 languages) |

## Commit conventions

| Prefix | When |
|--------|------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `test:` | Tests only |
| `refactor:` | Refactor without behaviour change |
| `chore:` | Build, tooling, dependencies |

---
[← Classification](classification.md) · [Home](index.md)
