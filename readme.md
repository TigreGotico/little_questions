# little_questions

Classify sentences by **type** (question, command, statement, exclamation, request)
and, for questions, by **expected answer category** (EAT taxonomy: 7 main, 53 fine-grained).

Both classifiers download their ONNX models automatically from HuggingFace on first use.

## Install

```bash
pip install little-questions
```

## Quick start

```python
from little_questions import Sentence

# A question
s = Sentence("Who invented the telephone?")
print(type(s).__name__)          # Question
print(s.sentence_type)           # question
print(s.classification)          # HUM:ind
print(s.main_label)              # HUM
print(s.secondary_label)         # ind
print(s.pretty_label)            # individual (Human)
print(s.confidence)              # 0.94

# A command
s = Sentence("Play some jazz music.")
print(type(s).__name__)          # Command
print(s.sentence_type)           # command

# A statement
s = Sentence("The sky is blue.")
print(type(s).__name__)          # Statement
print(s.sentence_type)           # statement
```

---

## Sentence type classification

`Sentence(text)` returns the appropriate subclass based on sentence type:

| Class | `sentence_type` | Example |
|-------|----------------|---------|
| `Question` | `question` | "What is the capital of France?" |
| `Statement` | `statement` | "The sky is blue." |
| `Command` | `command` | "Open the door." |
| `Request` | `request` | "Could you pass the salt?" |
| `Exclamation` | `exclamation` | "What a beautiful day!" |

`Request` subclasses `Command`, so `isinstance(s, Command)` is `True` for requests too.

Sentence-type classification uses ONNX models for **7 languages** (en, de, es, fr, it, nl, pt)
and locale-rule heuristics for **21 languages** as fallback.

```python
from little_questions import Sentence, Question, Command, Statement

s = Sentence("Could you help me?")
assert isinstance(s, Command)   # Request subclasses Command
assert s.is_request
assert s.is_command

# Non-question handling — sentence_type tells you what to do with it
if s.is_question:
    answer_type = s.main_label   # route to answer-type handler
elif s.is_command or s.is_request:
    pass  # route to action handler
elif s.is_statement:
    pass  # route to knowledge handler
```

---

## Question classification — EAT taxonomy

For questions, a **two-stage calibrated classifier** determines the expected answer type:

- **Stage 1** — `eat7_svm_cal`: predicts the main category (ABBR, BOOL, DESC, ENTY, HUM, LOC, NUM)
- **Stage 2** — `eat53_svm_cal`: scores all 53 fine-grained labels; labels outside the
  stage-1 category are masked and the remaining probabilities are renormalised

| Main | Description | Fine-grained subtypes |
|------|-------------|-----------------------|
| `ABBR` | Abbreviation | `abb`, `exp` |
| `BOOL` | Yes/No question | `yesno` |
| `DESC` | Description | `def`, `desc`, `manner`, `reason` |
| `ENTY` | Entity | `animal`, `body`, `color`, `food`, `product`, `sport`, `substance`, … (23 total) |
| `HUM` | Human | `ind`, `gr`, `title`, `desc` |
| `LOC` | Location | `city`, `country`, `state`, `mount`, `water`, … |
| `NUM` | Numeric | `date`, `money`, `dist`, `count`, `temp`, `speed`, … (13 total) |

### Calibrated confidence scores

Every `Sentence` exposes a calibrated probability distribution over all 53 labels:

```python
s = Sentence("When did World War II end?")
print(s.classification)          # NUM:date
print(s.confidence)              # 0.97
print(s.classification_scores)  # {"NUM:date": 0.97, "NUM:period": 0.02, ...}
# Values are true probabilities (Platt sigmoid): sum ≈ 1.0
```

---

## Supported languages

| Capability | Languages | Backend |
|------------|-----------|---------|
| Sentence-type (question / statement / command / …) | en, de, es, fr, it, nl, pt | ONNX model |
| Sentence-type (heuristic fallback) | ca, pl, ro, sv, cs, da, hu, tr, ru, uk, el, eu, gl, fa | Locale JSON rules |
| Question answer-type (EAT 53-class) | en | ONNX model (two-stage) |
| Question answer-type (heuristic fallback) | all of the above | Locale JSON rules |

Multilingual EAT models are planned — translate scripts are in `train/` for when
the EAT dataset is extended to other languages.

---

## API reference

```python
from little_questions import (
    Sentence,
    Question, Statement, Command, Request, Exclamation,
    get_classifier,   # → EatClassifier singleton for a language
    get_scorer,       # → SentenceTypeClassifier singleton for a language
)
from little_questions.classifiers import (
    clear_classifier_cache,    # flush all cached model instances
    list_supported_languages,  # languages with sentence-type ONNX models
)
from little_questions.constants import (
    EAT_LABELS_7,    # ["ABBR","BOOL","DESC","ENTY","HUM","LOC","NUM"]
    EAT_LABELS_53,   # full 53-label list
    SENTENCE_TYPES,  # ["command","exclamation","question","request","statement"]
    MAIN_LABEL_NAMES, SEC_LABEL_NAMES,
)
```

### `Sentence` attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `sentence_type` | `str` | `question` / `statement` / `command` / `request` / `exclamation` |
| `classification` | `str` | Full EAT label, e.g. `"HUM:ind"` |
| `main_label` | `str` | Main EAT category, e.g. `"HUM"` |
| `secondary_label` | `str \| None` | EAT sub-type, e.g. `"ind"` |
| `pretty_label` | `str` | Human-readable, e.g. `"individual (Human)"` |
| `classification_scores` | `dict[str, float]` | Calibrated probabilities over all 53 EAT labels |
| `confidence` | `float` | Max value from `classification_scores` |
| `lang` | `str` | Language code used at construction |

### Boolean properties

`is_question`, `is_statement`, `is_command`, `is_request`, `is_exclamation`

---

## Models

| Model | Size | Accuracy |
|-------|------|----------|
| `eat53_svm_cal_EN_0.9.0.onnx` | 16 MB | 91.0% macro F1 (53-class) |
| `eat7_svm_cal_EN_0.9.0.onnx` | 2.6 MB | 95.6% macro F1 (7-class) |
| Two-stage (default runtime) | 18.6 MB total | **93.4%** macro F1 (53-class) |
| `sentence_type_EN_0.8.0.onnx` | 2.3 MB | — |

Models are published at
[TigreGotico/eat-classifiers](https://huggingface.co/TigreGotico/eat-classifiers)
and [TigreGotico/sentence-types](https://huggingface.co/TigreGotico/sentence-types).

---

## Training

```bash
pip install little-questions[train]

python -m train.train_eat           # ONNX baselines (svm, logreg, sgd, svm_cal)
python -m train.train_eat_m2v       # Model2Vec variants (12 models)
python -m train.benchmark_eat       # Full benchmark + plots
python -m train.push_to_hf          # Push to TigreGotico/eat-classifiers
```

Training data: [TigreGotico/EAT](https://huggingface.co/datasets/TigreGotico/EAT) (30K EN questions, 53 labels)
and [TigreGotico/sentence-types-multilingual](https://huggingface.co/datasets/TigreGotico/sentence-types-multilingual) (80K multilingual samples).

---

## Dependencies

**Runtime:** `onnxruntime`, `numpy`, `huggingface_hub`

**Training:** `scikit-learn`, `skl2onnx`, `onnx`, `model2vec`, `mlflow`
