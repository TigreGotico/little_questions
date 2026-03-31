# Contributing & Development Guide

## Setup

```bash
git clone https://github.com/OpenJarbas/little_questions
cd little_questions
uv pip install -e ".[train]"
```

NLTK data required for English tests:
```bash
uv run python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

## Running tests

```bash
uv run pytest test/ -v --cov=little_questions --cov-report=term-missing
```

## Project layout

```
little_questions/
├── __init__.py          # Sentence, classify(), classify_batch(), subclasses
├── classifiers/
│   └── __init__.py      # Classifier (ONNX/joblib), get_classifier(), get_scorer()
├── sentence_type.py     # SentenceTypeClassifier (TF-IDF + LinearSVC)
├── models/
│   └── __init__.py      # download(), get_model_path(), LANG2MODEL, MODEL2URL, MODEL2SHA256
├── constants.py         # SUPPORTED_LANGUAGES, SENTENCE_TYPES
└── version.py           # version block

train/                   # Training-only (not installed with the package)
├── classifiers.py       # LinearSVCClassifier, Model2VecClassifier
├── baselines.py         # PunctuationScorer, HeuristicScorer (benchmarking only)
├── features.py          # LinguisticFeaturesTransformer (40+ POS/lexical features)
├── metrics.py           # EvalResult, evaluate(), compare()
├── mlflow_config.py     # MLflow tracking helpers
├── tune.py              # Optuna HPO + SGD per-epoch training
├── train_all.py         # Train all language SVM models
└── compare_classifiers.py  # Benchmark SVM vs Potion vs baselines
```

## Adding a language

1. Add URL and filename entries in `little_questions/models/__init__.py`:
   - `LANG2MODEL["<code>"]` and `LANG2MODEL["<code>_small"]`
   - `download_<code>()` function

2. Add to `little_questions/constants.py`:
   - `SUPPORTED_LANGUAGES`

3. Add a `LANG_CONFIG` entry in `train/compare_classifiers.py`.

4. Train the model (see Training section below).

5. Write smoke tests in `test/test_sentence.py`.

## Training a model

Training scripts live in `train/`.

```bash
# Set MLflow credentials (optional — runs log to localhost if unset)
export MLFLOW_TRACKING_URI=https://mlflow.example.com
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD=<token>

# Train all languages, 52-class, export ONNX
uv run python -m train.train_all

# Train a single language
uv run python -m train.train_en --classes 52

# Benchmark SVM vs Potion vs heuristics
uv run python -m train.compare_classifiers --lang en --classes 52

# Hyperparameter search
uv run python -m train.tune --lang en --classes 52 --n-trials 50
```

Trained ONNX models are saved to `~/.local/share/little_questions/` and uploaded to the configured MLflow run as artifacts.

## Commit conventions

Follow [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | When |
|--------|------|
| `feat:` | New feature or entry point |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `test:` | Tests only |
| `refactor:` | Refactor without behaviour change |
| `chore:` | Build, tooling, dependencies |

Always include:
- AI model name if AI-generated
- `Verified via: uv run pytest test/ -v` or equivalent
