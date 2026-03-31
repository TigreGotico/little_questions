# API Reference

## `little_questions` — top-level package

### `classify`

```python
classify(text: str, lang: str = "en") -> str
```
`little_questions/__init__.py:8`

Return the COSC label for *text* without sentence-type scoring. Faster than `Sentence.parse()` when only the label is needed.

### `classify_batch`

```python
classify_batch(texts: list[str], lang: str = "en") -> list[str]
```
`little_questions/__init__.py:25`

Classify multiple texts in a single model call.

---

### `Sentence`

```python
class Sentence(str)
```
`little_questions/__init__.py:99`

A classified sentence. Subclasses `str`, so all string operations work normally. The concrete
subclass (`Question`, `Command`, etc.) is chosen at construction time.

**Construction:**

```python
Sentence(content: str, model: str = "en", scorer: Optional[str] = None) -> Sentence
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `content` | — | The sentence text. |
| `model` | `"en"` | Language code for the COSC classifier. One of `en es pt ca fr de it nl`. |
| `scorer` | `None` | Language code for the sentence-type scorer. Defaults to `model`. |

Returns a concrete subclass instance (`Question`, `Command`, etc.).

**Classmethod:**

```python
Sentence.parse(text: str, lang: str = "en") -> Sentence
```
`little_questions/__init__.py:121`

Equivalent to `Sentence(text, model=lang)`.

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
| `main_label` | `str` | Top-level COSC category. | `__init__.py:179` |
| `secondary_label` | `str` | Fine-grained COSC subtype. | `__init__.py:184` |
| `pretty_label` | `str` | Human-readable combined label. | `__init__.py:189` |
| `is_exclamation` | `bool` | `True` when `isinstance(self, Exclamation)`. | `__init__.py:196` |
| `is_request` | `bool` | `True` when `isinstance(self, Request)`. | `__init__.py:201` |
| `is_statement` | `bool` | `True` when `isinstance(self, Statement)`. | `__init__.py:206` |
| `is_command` | `bool` | `True` when `isinstance(self, Command)`. | `__init__.py:211` |
| `is_question` | `bool` | `True` when `isinstance(self, Question)`. | `__init__.py:216` |

---

### Concrete sentence subclasses

All subclass `Sentence` (and `str`). `little_questions/__init__.py:221`

| Class | Inherits | Notes |
|-------|----------|-------|
| `Question` | `Sentence` | `sentence_type == "question"` |
| `Command` | `Sentence` | `sentence_type == "command"` |
| `Request` | `Command` | `sentence_type == "request"`. Also satisfies `is_command`. |
| `Statement` | `Sentence` | `sentence_type == "statement"` |
| `Exclamation` | `Sentence` | `sentence_type == "exclamation"` |

---

## `little_questions.classifiers`

`little_questions/classifiers/__init__.py`

### `get_classifier(model_id: str) -> Classifier`

Line 118. Returns a loaded `Classifier` for the given language code. Results are cached — the model file is read from disk only on the first call per language.

```python
from little_questions.classifiers import get_classifier
clf = get_classifier("en")
labels = clf.predict(["Who invented the telephone?"])  # ["HUM:ind"]
```

### `get_scorer(lang: Optional[str] = None) -> SentenceTypeClassifier`

Line 129. Returns the `SentenceTypeClassifier` for *lang*. When no trained model file is present, it falls back to the internal punctuation + first-word heuristic.

### `clear_classifier_cache() -> None`

Line 107. Clears the lazy-load cache. All subsequent `get_classifier()` calls will reload from disk.

### `list_supported_languages() -> List[str]`

Line 113. Returns the list of all supported language codes (reads from `SUPPORTED_LANGUAGES`).

---

### `Classifier`

`little_questions/classifiers/__init__.py:20`

COSC question classifier using ONNX Runtime inference. Falls back to joblib `.pkl` when given a `.pkl` path.

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(pipeline_id: str)` | `pipeline_id` is the language code, e.g. `"en"`. |
| `predict` | `(texts: List[str]) -> List[str]` | Return list of COSC labels. |
| `predict_proba` | `(texts: List[str]) -> List[List[float]]` | Probability estimates (sklearn only). |
| `load_from_file` | `(path: Optional[str])` | Load from path (or default model path). Detects format from extension. |

---

## `little_questions.models`

`little_questions/models/__init__.py`

### `get_model_path(model: str = "en") -> str`

Line 149. Resolve the filesystem path for a model. Downloads the model if not already cached.

| `model` value | Resolution |
|---------------|------------|
| An existing filesystem path | Returned as-is |
| A key in `LANG2MODEL` | Downloads if absent, returns path |
| Anything else | Raises `ValueError` |

### `download(model_id: str, force: bool = False) -> str`

Line 120. Download a model file to the XDG data directory and verify its SHA-256 checksum. `model_id` is a language code (e.g. `"en"`) or a key in `LANG2MODEL`. Skips if already downloaded unless `force=True`. Returns the absolute path.

### Language-specific download helpers

| Function | Line | Downloads |
|----------|------|-----------|
| `download_en()` | 178 | EN 52-class + 6-class models + NLTK data |
| `download_es()` | 192 | ES 52-class + 6-class models |
| `download_pt()` | 186 | PT 52-class + 6-class models |
| `download_ca()` | 216 | CA 52-class + 6-class models |
| `download_fr()` | 198 | FR 52-class + 6-class models |
| `download_de()` | 210 | DE 52-class + 6-class models |
| `download_it()` | 204 | IT 52-class + 6-class models |
| `download_nl()` | 222 | NL 52-class + 6-class models |

### `MODEL2URL`

Line 39. Dict mapping language code (e.g. `"en"`) to the GitHub release download URL.

### `LANG2MODEL`

Line 20. Dict mapping language code (e.g. `"en"`) to the expected model filename. Includes `_small` variants (6-class).

### `MODEL2SHA256`

Line 46. Dict mapping language code to expected SHA-256 digest. Entries are `None` until populated after a release upload; missing entries log a warning and skip verification.
