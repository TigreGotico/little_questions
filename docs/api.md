# API Reference

## `little_questions` (top-level package)

### `Sentence`

```python
class Sentence(str)
```

A classified sentence. Subclasses `str`, so all string operations work normally. The concrete subclass (`Question`, `Statement`, `Command`, `Request`, `Exclamation`) is chosen automatically at construction time.

**Construction:**

```python
Sentence(content: str, lang: str = "en") -> Sentence
```

Returns a concrete subclass. Raises `RuntimeError` if no ONNX model is available for the requested language.

**Class method:**

```python
Sentence.parse(text: str, lang: str = "en") -> Sentence
```

Equivalent to `Sentence(text, lang)`.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `sentence_type` | `str` | `question` / `statement` / `command` / `request` / `exclamation` |
| `classification` | `str` | Full EAT label, e.g. `"HUM:ind"` |
| `classification_scores` | `dict[str, float]` | Calibrated probabilities over all 53 EAT labels (sum ≈ 1.0) |
| `confidence` | `float` | Max value from `classification_scores` |
| `lang` | `str` | Language code used at construction |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `main_label` | `str` | Top-level EAT category (ABBR/BOOL/DESC/ENTY/HUM/LOC/NUM) |
| `secondary_label` | `str \| None` | EAT sub-type, or `None` |
| `pretty_label` | `str` | Human-readable, e.g. `"individual (Human)"` |
| `is_question` | `bool` | `sentence_type == "question"` |
| `is_statement` | `bool` | `sentence_type == "statement"` |
| `is_command` | `bool` | `sentence_type in ("command", "request")` |
| `is_request` | `bool` | `sentence_type == "request"` |
| `is_exclamation` | `bool` | `sentence_type == "exclamation"` |

---

### Concrete subclasses

All subclass `Sentence` (and `str`).

| Class | `sentence_type` | Notes |
|-------|----------------|-------|
| `Question` | `question` | |
| `Statement` | `statement` | Adds `answer_polarity`, see below |
| `Command` | `command` | |
| `Request` | `request` | Subclass of `Command`. Also satisfies `is_command` |
| `Exclamation` | `exclamation` | |

---

### `Statement`: answer polarity

`Statement` adds three lazy-loaded properties for classifying yes/no responses. The yes/no ONNX model downloads on first access. A `RuntimeError` is raised if no model is available.

| Property | Type | Description |
|----------|------|-------------|
| `answer_polarity` | `str` | `"yes"`, `"no"`, or `"maybe"` |
| `answer_polarity_scores` | `dict[str, float]` | Calibrated probabilities over `yes`/`no`/`maybe` |
| `is_affirmative` | `bool` | `answer_polarity == "yes"` |
| `is_negative` | `bool` | `answer_polarity == "no"` |

```python
q = Sentence("Is the sky blue?")   # → Question, classification BOOL:yesno
a = Sentence("Yes, absolutely.")   # → Statement
print(a.answer_polarity)           # yes
print(a.is_affirmative)            # True
```

---

### Helper functions

```python
get_classifier(lang: str = "en") -> EatClassifier
```
Return the `EatClassifier` singleton for *lang*.

```python
get_scorer(lang: str = "en") -> SentenceTypeClassifier
```
Return the `SentenceTypeClassifier` singleton for *lang*.

```python
get_yesno_classifier(lang: str = "en") -> YesNoClassifier
```
Return the `YesNoClassifier` singleton for *lang*.

---

## `little_questions.classifiers`

### `EatClassifier`

Two-stage calibrated EAT classifier. Stage 1 (`eat7_svm_cal`) predicts the main category. Stage 2 (`eat53_svm_cal`) scores all 53 labels and renormalises within the predicted main category.

```python
EatClassifier.get_instance(lang: str) -> EatClassifier
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `predict` | `(text: str) -> str` | Top EAT label, e.g. `"HUM:ind"` |
| `score` | `(text: str) -> dict[str, float]` | Calibrated probabilities over all 53 labels |

Raises `RuntimeError` if no model is available for *lang*.

---

### `SentenceTypeClassifier`

ONNX sentence-type classifier. Auto-downloads from `TigreGotico/sentence-types`.

```python
SentenceTypeClassifier.get_instance(lang: str) -> SentenceTypeClassifier
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `predict` | `(text: str) -> str` | Sentence type: `question`/`statement`/`command`/`request`/`exclamation` |
| `score` | `(text: str) -> dict[str, float]` | Softmax probabilities over sentence types |

Supported languages for ONNX inference: `en`, `de`, `es`, `fr`, `it`, `nl`, `pt`. Raises `RuntimeError` for unsupported languages.

---

### `YesNoClassifier`

ONNX yes/no polarity classifier. Auto-downloads from `TigreGotico/yes-no-classifiers`.

```python
YesNoClassifier.get_instance(lang: str) -> YesNoClassifier
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `predict` | `(text: str) -> str` | `"yes"`, `"no"`, or `"maybe"` |
| `score` | `(text: str) -> dict[str, float]` | Calibrated probabilities over `yes`/`no`/`maybe` |

Tries a language-specific model first, then falls back to the multilingual model. Raises `RuntimeError` if neither is available.

---

### Cache utilities

```python
from little_questions.classifiers import clear_classifier_cache, list_supported_languages
clear_classifier_cache()        # force reload on next use
list_supported_languages()      # languages with a sentence-type ONNX model
```

---

## `little_questions.constants`

```python
from little_questions.constants import (
    EAT_LABELS_7,       # ["ABBR","BOOL","DESC","ENTY","HUM","LOC","NUM"]
    EAT_LABELS_53,      # full 53-label list
    SENTENCE_TYPES,     # ["command","exclamation","question","request","statement"]
    MAIN_LABEL_NAMES,   # {"ABBR": "Abbreviation", "BOOL": "Boolean", ...}
    SEC_LABEL_NAMES,    # {"yesno": "yes/no question", "ind": "individual", ...}
)
```

---

## `little_questions.models`

```python
from little_questions.models import (
    get_eat_model_path,           # (filename) -> str | None
    get_sentence_type_model_path, # (lang) -> str | None
    get_yesno_model_path,         # (lang, version) -> str | None
)
```

Each function downloads the model from HuggingFace on first call and returns the local cache path. Returns `None` if the download fails.

HuggingFace repos:
- EAT models: `TigreGotico/eat-classifiers`
- Sentence-type models: `TigreGotico/sentence-types`
- Yes/no models: `TigreGotico/yes-no-classifiers`

---
[Home](index.md) · [Models →](models.md)
