# SUGGESTIONS — little_questions

Agent-generated improvement proposals. Each item is scoped, actionable, and non-breaking.

---

## Open

### S-003 — Stream model downloads with progress

**Priority**: Low
**Effort**: Small

Replace the blocking `requests.get(url, timeout=300).content` in `download()` with a
streaming download using `iter_content(chunk_size=8192)`. Add at least one retry on
transient failures. File: `little_questions/models/__init__.py:download`.

---

### S-004 — Add SHA-256 checksum verification for downloaded models

**Priority**: Medium (security)
**Effort**: Small

Add a `MODEL2SHA256` dict mapping model ID to expected hex digest. After download,
verify with `hashlib.sha256`. Refuse to load if mismatch. Protects against corrupted
downloads and compromised release assets. File: `little_questions/models/__init__.py`.

---

### S-005 — Add `questions_only` fast path

**Priority**: Low
**Effort**: Small

When `Sentence` is used purely for COSC classification, instantiating the sentence-type
scorer (which imports NLTK and tokenises the input) wastes ~10 ms per call. A
`classify(text, lang)` function returning only `(main_label, secondary_label)` would
benefit high-throughput pipelines. File: `little_questions/__init__.py`.

---

### S-008 — Publish sentence-type accuracy per language

**Priority**: Medium
**Effort**: Medium

Sentence-type detection for non-English languages uses punctuation heuristics with no
published accuracy. Add a `scripts/eval_scorer.py` that runs against a hand-labelled
evaluation set and prints a confusion matrix. Publish results in `docs/classification.md`.

---

### S-009 — Fix logic bug: `or` should be `and` in `exclamation_score`

**Priority**: Critical (correctness bug)
**Effort**: Trivial

`little_questions/classifiers/legacy.py:305–308`:
```python
elif not text.lower().startswith("what a ") or not text.lower().startswith("what an "):
    score -= 0.1
```
`not A or not B` is always `True` (De Morgan: equivalent to `not (A and B)`), so the
`score -= 0.1` penalty fires unconditionally. The intent is to penalise sentences that
start with *neither* "what a " nor "what an ":
```python
elif not (text.lower().startswith("what a ") or text.lower().startswith("what an ")):
    score -= 0.1
```

---

### S-010 — Fix operator precedence: `not x in y` → `x not in y`

**Priority**: High (correctness)
**Effort**: Small (search-and-replace)

`little_questions/classifiers/legacy.py:135,137,139,141,303`:
```python
if not tokens[-1] in last_tokens:   # parsed as: (not tokens[-1]) in last_tokens
if not tokens[0]  in first_tokens:  # same bug
if not tagged[0][1] in start_pos_tags:
if not tagged[-1][1] in end_pos_tags:
if not text.split(" ")[0].lower() in first_tokens:
```
Python parses `not x in y` correctly as `x not in y` due to how `not in` is
special-cased — so this is not a runtime bug, but it is a **misleading anti-pattern**
that every reader must mentally re-parse. Replace all instances with `x not in y`.

---

### S-011 — Eliminate `SentenceScorer`/`Classifier` duplication between `base.py` and `classifiers/__init__.py`

**Priority**: High (maintenance)
**Effort**: Medium

Both files define nearly identical `SentenceScorer` and `Classifier` classes:
- `little_questions/classifiers/base.py` — typed version with full docstrings
- `little_questions/classifiers/__init__.py` — slightly different ONNX loading, fewer type hints

The `__init__.py` version also loads classes from a companion `.pkl` sidecar file
next to the `.onnx` to extract `classes_`, while `base.py` tries to read them from
the ONNX graph. Neither approach is documented.

**Fix**: Delete the duplicate classes from `classifiers/__init__.py` and import from
`base.py`. Decide on a single strategy for extracting class labels from ONNX models.

---

### S-012 — Remove `get_scorer()` triplicate

**Priority**: Medium
**Effort**: Small

`get_scorer()` is defined in three places with different behaviour:
1. `little_questions/classifiers/__init__.py:173` — EN uses trained classifier, others use `SentenceScorer()`
2. `little_questions/classifiers/lang/__init__.py:15` — EN uses trained classifier, others use `SentenceScorer()`
3. `little_questions/classifiers/lang/en/__init__.py:6` — always returns EN trained classifier

Only one canonical `get_scorer(lang)` is needed. Delete the other two.

---

### S-013 — Replace `SentenceScorer` static-only class with module-level functions

**Priority**: Medium (design smell)
**Effort**: Small

`little_questions/classifiers/base.py:SentenceScorer` and
`little_questions/classifiers/__init__.py:SentenceScorer` are collections of `@staticmethod`
methods with no instance state — namespace misuse. Convert to module-level functions in a
`scorers.py` module. The public API surface shrinks and the code becomes easier to import
and test.

Same applies to `SentenceScorerHeuristic` in `legacy.py` — all 7 public methods plus
`_score()` are `@staticmethod`.

---

### S-014 — Replace `pretty_label` if/elif chain with a dict lookup

**Priority**: Medium
**Effort**: Trivial

`little_questions/__init__.py:139–187` contains 50+ lines of nested if/elif to map
COSC codes to human-readable strings:
```python
if self.main_label == "ENTY":
    pretty_main = "Entity"
elif self.main_label == "DESC":
    ...
```

Replace with two module-level dicts:
```python
_MAIN_LABEL_NAMES = {"HUM": "Human", "ENTY": "Entity", "DESC": "Description",
                     "NUM": "Numeric", "LOC": "Location", "ABBR": "Abbreviation"}
_SEC_LABEL_NAMES  = {"def": "definition", "desc": "description", "ind": "individual", ...}
```
Then `pretty_label` becomes two dict lookups. Trivial to extend, trivial to test.

---

### S-015 — Remove dead `classifiers/lang/` sub-packages

**Priority**: Medium (clarity)
**Effort**: Small

Six of eight language sub-packages contain only empty `__init__.py` files:
`ca/__init__.py`, `de/__init__.py`, `es/__init__.py`, `fr/__init__.py`,
`it/__init__.py`, `pt/__init__.py` (all 0 lines). They imply language-specific
scorers that don't exist. The `postag.py` files in `ca/`, `es/`, `pt/` are never
imported by anything in the inference package.

**Fix**: Delete empty `__init__.py` files. Either delete the `postag.py` files too,
or move them to `train/` where they belong (they are only needed during training).

---

### S-016 — Consolidate duplicate constant lists

**Priority**: Low
**Effort**: Trivial

`SENTENCE_TYPES` and `SUPPORTED_LANGUAGES` are defined twice:
- `little_questions/sentence_type.py:16,18`
- `little_questions/classifiers/__init__.py:191`

Create `little_questions/constants.py` and import from there.

---

### S-017 — Fix duplicate entry in `YES_NO_STARTERS`

**Priority**: Low (data correctness)
**Effort**: Trivial

`little_questions/classifiers/legacy.py:26–27`:
```python
"have",
"has",   # appears at line 21 and again here
```
`"has"` appears twice. Use a `frozenset` literal instead of a `list` to make
duplicates impossible and membership tests O(1).

---

### S-018 — Tighten `Optional[Any]` type hints

**Priority**: Low
**Effort**: Small

`little_questions/sentence_type.py:32–34` declares `Optional[Any]` for three sklearn
objects, defeating the type checker. Same pattern in `classifiers/base.py:89–91` and
`classifiers/__init__.py:93–95`. Use `TYPE_CHECKING` imports:
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sklearn.svm import LinearSVC
    from sklearn.feature_extraction.text import TfidfVectorizer
    import onnxruntime as ort
```

---

### S-019 — Thread-safe classifier caches

**Priority**: Low (correctness under concurrency)
**Effort**: Small

`_LAZY_LOADING` (module dict in `classifiers/__init__.py:188`) and
`SentenceTypeClassifier._instances` (class dict in `sentence_type.py:28`) are
written without locks. Concurrent first-use calls could load the same model twice or
corrupt the dict. Wrap writes with `threading.Lock`.

---

### S-020 — `Sentence.__new__` scorer language extraction is a TODO

**Priority**: Low
**Effort**: Trivial

`little_questions/__init__.py:90–93`:
```python
if model.startswith("http"):
    # TODO naming convention to extract lang
    if "en" in model:
        scorer = "en"
```
The TODO has no path to resolution and silently defaults to `"en"` for any HTTP URL
containing the string `"en"`. Either implement a naming convention or remove HTTP
URL support from `model` entirely (it raises `NotImplementedError` in
`get_model_path()` anyway, so HTTP URLs are already unsupported end-to-end).

---

### S-021 — Train scripts: extract shared `load_data()` to `train/utils.py`

**Priority**: Low
**Effort**: Trivial

`load_data()` is duplicated across train scripts. Extract to `train/utils.py` and
import from there.

---

## Completed (post v0.8.0 refactor)

| ID | Suggestion | Resolved |
|----|-----------|---------|
| S-009 | Fix `or` logic bug in `exclamation_score` | `legacy.py:305` |
| S-010 | Fix `not x in y` anti-pattern | `legacy.py:135,137,139,141,303,383` |
| S-011 | Eliminate `SentenceScorer`/`Classifier` duplication | `base.py` canonical; dead `Classifier` in `base.py` removed |
| S-012 | Remove `get_scorer()` triplicate | Single definition in `classifiers/__init__.py` |
| S-014 | Replace `pretty_label` if/elif chain | `__init__.py:_MAIN_LABEL_NAMES`, `_SEC_LABEL_NAMES` |
| S-015 | Remove dead lang sub-packages | Deleted `classifiers/lang/{ca,de,es,fr,it,nl,pt}/`; postag files moved to `train/lang/` |
| S-016 | Consolidate duplicate constants | `little_questions/constants.py` |
| S-017 | Fix duplicate `"has"` in `YES_NO_STARTERS` | Converted to `frozenset` |
| S-020 | Remove dead HTTP TODO branch | Removed from `Sentence.__new__` |
| S-021 | Extract `load_data()` to `train/utils.py` | `train/utils.py:load_data` |

## Completed (v0.8.0)

| ID | Suggestion | Resolved |
|----|-----------|---------|
| S-001 | Add `Sentence.parse()` classmethod | `little_questions/__init__.py:Sentence.parse` |
| S-002 | Add `clear_classifier_cache()` and `list_supported_languages()` | `little_questions/classifiers/__init__.py` |
| S-006 | Replace `pickle.load()` with `joblib.load()` in `load_model()` | `load_model()` removed; `Classifier.load_from_file()` uses ONNX/joblib |
| S-007 | Add trove classifiers and project URLs to `pyproject.toml` | Done in v0.8.0 |
