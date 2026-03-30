# Contributing & Development Guide

## Setup

```bash
git clone https://github.com/OpenJarbas/little_questions
cd little_questions
uv pip install -e ".[dev]"
```

NLTK data required for the English scorer tests:
```bash
uv run python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

## Running tests

```bash
uv run pytest test/ -v --cov=little_questions --cov-report=term-missing
```

Tests run without network access or model files. `test/conftest.py` stubs `JarbasModelZoo`
and `xdg.BaseDirectory` so CI environments do not need those packages installed.

## Project layout

```
little_questions/
├── __init__.py                  # Sentence, Question, Command, etc.
├── classifiers/
│   ├── __init__.py              # get_classifier(), get_scorer()
│   ├── base.py                  # SentenceScorer, Classifier, *TextClassifier
│   ├── features.py              # WordFeaturesTransformer, WordFeaturesVectorizer
│   └── lang/
│       ├── __init__.py          # get_pipeline(lang) dispatcher
│       ├── en/                  # SentenceScorerEN, English feature pipeline
│       ├── es/ pt/ ca/ fr/ de/ it/
└── models/
    └── __init__.py              # download(), get_model_path(), LANG2MODEL
```

## Adding a language

1. Create `little_questions/classifiers/lang/<code>/` with:
   - `__init__.py` — expose `get_pipeline_<code>()` returning a `FeatureUnion`
   - `features.py` — language-specific `LemmatizerTransformer` or `POSTaggerVectorizer`

2. Register in `little_questions/classifiers/lang/__init__.py`:
   ```python
   from little_questions.classifiers.lang.<code> import get_pipeline_<code>
   _PIPELINES["<code>"] = get_pipeline_<code>
   ```

3. Add to `little_questions/models/__init__.py`:
   - URL entries in `MODEL2URL`
   - Path entries in `LANG2MODEL`
   - `download_<code>()` function

4. Train the model (see Training section below).

5. Write smoke tests in `test/test_sentence.py`.

## Training a model

Training scripts live in `train_scripts/`. Each language follows the same pattern:

```
train_scripts/
├── clean.py                   # Shared: clean and normalise raw data
├── train_en.py                # English training script
├── train_es.py                # Spanish
├── clean_data/                # Preprocessed training data (CSV)
│   ├── questions_en_train.csv
│   └── ...
└── reports/                   # Accuracy reports from last training run
```

**Workflow:**

```bash
# 1. Clean raw data (only needed when adding new training examples)
uv run python train_scripts/clean.py

# 2. Train a specific language
uv run python train_scripts/train_en.py
# Produces: questions52_svm_EN_<version>.pkl  (in current directory)

# 3. Move the model to the XDG cache directory
mv questions52_svm_EN_*.pkl ~/.local/share/little_questions/

# 4. Test that the model loads
uv run python -c "from little_questions import Sentence; print(Sentence('Who are you?'))"
```

The default classifier is `LinearSVCTextClassifier`. To experiment with other algorithms,
edit the `train_<lang>.py` script and replace the classifier class. Compare accuracy reports
in `train_scripts/reports/`.

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

## Release process

See `MAINTAINERS_GUIDE.md`.
