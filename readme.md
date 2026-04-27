# little_questions

Classify English sentences by type and expected answer category — powered by
calibrated ONNX models trained on the
[EAT dataset](https://huggingface.co/datasets/TigreGotico/EAT).

Models download automatically from
[TigreGotico/eat-classifiers](https://huggingface.co/TigreGotico/eat-classifiers)
on first use.

## Install

```bash
pip install little-questions
```

## Quick start

```python
from little_questions import Sentence

s = Sentence("Who invented the telephone?")

print(type(s).__name__)          # Question
print(s.classification)          # HUM:ind
print(s.main_label)              # HUM
print(s.secondary_label)         # ind
print(s.pretty_label)            # individual (Human)
print(s.confidence)              # 0.94
print(s.sentence_type)           # question
```

## Sentence types

`Sentence(text)` returns the appropriate subclass automatically:

| Class | sentence_type | Example |
|-------|--------------|---------|
| `Question` | `question` | "What is the capital of France?" |
| `Statement` | `statement` | "The sky is blue." |
| `Command` | `command` | "Open the door." |
| `Request` | `request` | "Could you pass the salt?" |
| `Exclamation` | `exclamation` | "What a beautiful day!" |

`Request` subclasses `Command`, so `isinstance(s, Command)` is `True` for both.

## Question classification — EAT taxonomy

Questions are classified into **7 main categories** and **53 fine-grained subtypes**:

| Main | Description | Subtypes |
|------|-------------|----------|
| `ABBR` | Abbreviation | `abb`, `exp` |
| `BOOL` | Yes/No question | `yesno` |
| `DESC` | Description | `def`, `desc`, `manner`, `reason` |
| `ENTY` | Entity | `animal`, `body`, `color`, `food`, `product`, `sport`, `substance`, … (23 total) |
| `HUM` | Human | `ind`, `gr`, `title`, `desc` |
| `LOC` | Location | `city`, `country`, `state`, `mount`, `water`, … |
| `NUM` | Numeric | `date`, `money`, `dist`, `count`, `temp`, `speed`, … (13 total) |

```python
from little_questions import Sentence

examples = [
    "Is Paris in France?",
    "Who wrote Hamlet?",
    "What does NASA stand for?",
    "How fast does light travel?",
    "Where is Mount Everest?",
]
for text in examples:
    s = Sentence(text)
    print(f"{s.classification:<15} {s.confidence:.2f}  {text}")
```

## Confidence scores

Every `Sentence` exposes calibrated probabilities over all 53 labels:

```python
s = Sentence("When did World War II end?")
print(s.confidence)              # max probability, e.g. 0.97
print(s.classification_scores)  # {"NUM:date": 0.97, "NUM:period": 0.02, ...}
```

Output[1] of the calibrated ONNX model is a true probability vector
(Platt sigmoid, values in [0, 1], sum ≈ 1.0).

## Supported languages

Sentence-type classification (question / statement / command / …) is
available for:

`en` `de` `es` `fr` `it` `nl` `pt`

Question-type classification (EAT taxonomy) is English-only in this release.
Multilingual EAT models are planned.

Heuristic fallbacks (locale JSON rules) cover 21 additional languages when
ONNX models are unavailable.

## API reference

```python
from little_questions import (
    Sentence,
    Question, Statement, Command, Request, Exclamation,
    get_classifier,   # → EatClassifier singleton for a language
    get_scorer,       # → SentenceTypeClassifier singleton for a language
)
from little_questions.classifiers import (
    clear_classifier_cache,    # flush cached model instances
    list_supported_languages,  # languages with sentence-type ONNX models
)
from little_questions.constants import (
    EAT_LABELS_7,    # ["ABBR","BOOL","DESC","ENTY","HUM","LOC","NUM"]
    EAT_LABELS_53,   # full 53-label list
    MAIN_LABEL_NAMES, SEC_LABEL_NAMES,  # human-readable name dicts
)
```

### `Sentence` attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `classification` | `str` | Full EAT label, e.g. `"HUM:ind"` |
| `main_label` | `str` | Main category, e.g. `"HUM"` |
| `secondary_label` | `str \| None` | Sub-type, e.g. `"ind"` |
| `pretty_label` | `str` | Human-readable, e.g. `"individual (Human)"` |
| `classification_scores` | `dict[str, float]` | Calibrated probabilities over all 53 labels |
| `confidence` | `float` | Max value from `classification_scores` |
| `sentence_type` | `str` | `question` / `statement` / `command` / `request` / `exclamation` |
| `lang` | `str` | Language code used at construction |

## Training

Training requires the `[train]` extra:

```bash
pip install little-questions[train]
```

```bash
# Train all ONNX baselines (svm, logreg, sgd, svm_cal) at 53 and 7 classes
python -m train.train_eat

# Train Model2Vec variants (potion-base 2M/8M/32M × plain/+tfidf × 53c/7c)
python -m train.train_eat_m2v

# Run full benchmark and regenerate plots
python -m train.benchmark_eat

# Push all models to HuggingFace
python -m train.push_to_hf
```

All models are trained on [TigreGotico/EAT](https://huggingface.co/datasets/TigreGotico/EAT)
and published to [TigreGotico/eat-classifiers](https://huggingface.co/TigreGotico/eat-classifiers).

## Models

| Model | Size | 53-class macro F1 |
|-------|------|-------------------|
| `eat53_svm_cal_EN_0.9.0.onnx` *(default)* | 16 MB | 0.909 |
| `eat53_svm_EN_0.9.0.onnx` | 41 MB | 0.915 |
| `m2v-potion-base-32M+tfidf-en-53c` | — | 0.921 |
| two-stage `svm_cal` *(benchmark only)* | — | 0.934 |

## Datasets

Two datasets were built for this project:

- [TigreGotico/EAT](https://huggingface.co/datasets/TigreGotico/EAT) —
  Expected Answer Type, 30K English questions, 53 fine-grained labels
- [TigreGotico/sentence-types-multilingual](https://huggingface.co/datasets/TigreGotico/sentence-types-multilingual) —
  Sentence type, 80K multilingual samples

## Dependencies

**Runtime:** `onnxruntime`, `numpy`, `huggingface_hub`

**Training:** `scikit-learn`, `skl2onnx`, `onnx`, `model2vec`, `mlflow`
