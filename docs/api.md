# API Reference

## `little_questions` — top-level package

### `Sentence`

```python
class Sentence(str)
```
`little_questions/__init__.py:18`

A classified sentence. Subclasses `str`, so all string operations work normally. The concrete
subclass (`Question`, `Command`, etc.) is chosen at construction time.

**Construction:**

```python
Sentence(content: str, model: str = "en", scorer: Optional[str] = None) -> Sentence
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `content` | — | The sentence text. |
| `model` | `"en"` | Language code for the COSC classifier. One of `en es pt ca fr de it`. |
| `scorer` | `None` | Language code for the sentence-type scorer. Defaults to `model`. |

Returns a concrete subclass instance (`Question`, `Command`, etc.).

**Alternative entry point (Phase 2+):**

```python
Sentence.parse(text: str, lang: str = "en") -> Sentence
```

Classmethod. Equivalent to `Sentence(text, model=lang)` but more discoverable.

**Attributes set at construction:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `classification` | `str` | Full COSC label, e.g. `"HUM:ind"`. |
| `model` | `str` | Language code used for COSC classification. |
| `sentence_type` | `str` | One of `question command statement exclamation request`. |
| `score` | `dict` | Per-type confidence scores from the sentence scorer. |

**Read-only properties:**

| Property | Type | Description | Source |
|----------|------|-------------|--------|
| `main_label` | `str` | Top-level COSC category. | `__init__.py:86` |
| `secondary_label` | `str` | Fine-grained COSC subtype. | `__init__.py:92` |
| `pretty_label` | `str` | Human-readable combined label. | `__init__.py:97` |
| `is_question` | `bool` | `True` when `isinstance(self, Question)`. | `__init__.py:168` |
| `is_command` | `bool` | `True` when `isinstance(self, Command)`. | `__init__.py:163` |
| `is_request` | `bool` | `True` when `isinstance(self, Request)`. | `__init__.py:153` |
| `is_statement` | `bool` | `True` when `isinstance(self, Statement)`. | `__init__.py:158` |
| `is_exclamation` | `bool` | `True` when `isinstance(self, Exclamation)`. | `__init__.py:147` |

---

### Concrete sentence subclasses

All subclass `Sentence` (and `str`).

| Class | Inherits | Notes |
|-------|----------|-------|
| `Question` | `Sentence` | `sentence_type == "question"` |
| `Command` | `Sentence` | `sentence_type == "command"` |
| `Request` | `Command` | `sentence_type == "request"`. Also satisfies `is_command`. |
| `Statement` | `Sentence` | `sentence_type == "statement"` |
| `Exclamation` | `Sentence` | `sentence_type == "exclamation"` |

---

## `little_questions.classifiers`

### `get_classifier(model_id: str) -> Classifier`

`little_questions/classifiers/__init__.py:16`

Returns a loaded `Classifier` for the given language code. Results are cached in
`_LAZY_LOADING` — the model file is read from disk only on the first call per language.

```python
from little_questions.classifiers import get_classifier
clf = get_classifier("en")
labels = clf.predict(["Who invented the telephone?"])  # ["HUM:ind"]
```

### `get_scorer(lang: Optional[str] = None) -> SentenceScorer`

`little_questions/classifiers/__init__.py:8`

Returns `SentenceScorerEN()` if `lang` starts with `"en"`, otherwise `SentenceScorer()`.

### `clear_classifier_cache() -> None`

`little_questions/classifiers/__init__.py` (Phase 2)

Clears the lazy-load cache. All subsequent `get_classifier()` calls will reload from disk.

### `list_supported_languages() -> List[str]`

`little_questions/classifiers/__init__.py` (Phase 2)

Returns `["en", "es", "pt", "ca", "fr", "de", "it"]`.

---

## `little_questions.classifiers.base`

### `SentenceScorer`

`little_questions/classifiers/base.py:22`

Rule-based sentence-type scorer. Language-agnostic fallback using terminal punctuation only.

| Method | Signature | Description |
|--------|-----------|-------------|
| `predict` | `(text: str) -> str` | Return best sentence type. |
| `score` | `(text: str) -> Dict[str, float]` | Return all type scores. |
| `question_score` | `(text: str) -> float` | Heuristic: 0.8 if ends `?`, else 0.4. |
| `statement_score` | `(text: str) -> float` | Heuristic: 0.5 if ends `.`, else 0. |
| `exclamation_score` | `(text: str) -> float` | Heuristic: 0.6 if ends `!`, else 0. |
| `command_score` | `(text: str) -> float` | Heuristic: 0.6 if ends `.`, 0.5 if `!`, else 0. |
| `request_score` | `(text: str) -> float` | Heuristic: 0.5 if ends `.` or `?`, else 0. |

### `Classifier`

`little_questions/classifiers/base.py:87`

Base class for COSC classifiers backed by a joblib-serialised sklearn pipeline.

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(pipeline_id: str)` | `pipeline_id` is the language code, e.g. `"en"`. |
| `train` | `(train_data, target_data)` | Fit the pipeline (subclasses implement). |
| `predict` | `(text: List[str]) -> List[str]` | Return list of COSC labels. |
| `save` | `(path: str)` | Persist the fitted pipeline via joblib. |
| `load_from_file` | `(path: Optional[str])` | Load from path (or default model path). |

**Available concrete subclasses** (all in `little_questions/classifiers/base.py`):

- `LinearSVCTextClassifier` — default; best accuracy on UIUC QC
- `LogRegTextClassifier`
- `RandomForestTextClassifier`
- `NaiveBayesTextClassifier`
- `PassiveAggressiveTextClassifier`
- `SGDTextClassifier`
- `PerceptronTextClassifier`

---

## `little_questions.classifiers.lang.en`

### `SentenceScorerEN`

`little_questions/classifiers/lang/en/__init__.py:23`

English sentence-type scorer using NLTK POS tags. More accurate than the base `SentenceScorer`.

Each `*_score(text)` static method runs `word_tokenize` + `pos_tag` via NLTK, then computes
a score ∈ [0, 1] based on start/end tokens, POS tag patterns, and unlikely-word penalties.
The internal `_score()` method at line 41 implements the shared scoring framework.

---

## `little_questions.models`

### `get_model_path(model: str) -> str`

`little_questions/models/__init__.py:163`

Resolve the filesystem path for a model. Downloads the model and required NLTK data if not present.

| `model` value | Resolution |
|---------------|------------|
| Starts with `http` | Raises `NotImplementedError` |
| A valid filesystem path | Returned as-is |
| A key in `LANG2MODEL` | Downloads if absent, returns path |
| Anything else | Raises `ValueError` |

### `download(model_id: str, force: bool = False) -> Optional[str]`

`little_questions/models/__init__.py:48`

Download a model file to the XDG data directory. `model_id` is a key in `MODEL2URL` (e.g.
`"questions52_EN"`). Skips if already downloaded unless `force=True`.

### Language-specific download helpers

| Function | Downloads |
|----------|-----------|
| `download_en()` | EN 52-class + 6-class models + NLTK data |
| `download_es()` | ES models + Spanish Brill tagger (via JarbasModelZoo if available) |
| `download_pt()` | PT models + Floresta tagger |
| `download_ca()` | CA models + Catalan Brill tagger |
| `download_fr()` | FR models |
| `download_de()` | DE models |
| `download_it()` | IT models |

### `MODEL2URL`

`little_questions/models/__init__.py:16`

Dict mapping model ID (e.g. `"questions52_EN"`) to GitHub release download URL.

### `LANG2MODEL`

`little_questions/models/__init__.py:79`

Dict mapping language code (e.g. `"en"`) to the expected filesystem path of the model file.
Includes `_small` variants (6-class) and `_tagger` variants (POS tagger, where applicable).
