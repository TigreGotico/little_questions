# AUDIT — little_questions

Known issues, technical debt, and security notes. Updated 2026-03-30.

---

## Critical bugs

### BUG-001 — Missing `.pkl` extension in `LANG2MODEL` for ca, fr, it, de
**File**: `little_questions/models/__init__.py:91–107`
**Severity**: High — all 4 languages fail to load at runtime

`LANG2MODEL` entries for `ca`, `fr`, `it`, `de` reference paths without the `.pkl` extension:
```python
"ca": join(..., "questions52_svm_CA_googtx_0.7.0a1"),   # missing .pkl
"fr": join(..., "questions52_svm_FR_googtx_0.7.0a1"),   # missing .pkl
"it": join(..., "questions52_svm_IT_googtx_0.7.0a1"),   # missing .pkl
"de": join(..., "questions52_svm_DE_googtx_0.7.0a1"),   # missing .pkl
```
`download()` saves files as `<id>.pkl` but `get_model_path()` returns the path without the extension, causing `joblib.load()` to raise `FileNotFoundError`. Fix: append `.pkl` to those four entries.

---

## Technical debt

### TD-001 — `SentenceScorerEN` lacks type hints
**File**: `little_questions/classifiers/lang/en/__init__.py:23–347`
All static methods (`predict`, `score`, `_score`, `question_score`, etc.) have no type annotations. Low priority since the class is internal, but inconsistent with the rest of the codebase post-Phase 1 refactor.

### TD-002 — `get_classifier()` undocumented cache
**File**: `little_questions/classifiers/__init__.py:5–24`
`_LAZY_LOADING` global dict caches loaded classifiers but is not documented. There is no `clear_classifier_cache()` or `list_supported_languages()` utility. Downstream code cannot reset the cache for memory management or test isolation.

### TD-003 — `Sentence` constructor is the only entry point
**File**: `little_questions/__init__.py:39`
`Sentence("text", model="en")` is the only callable entry point, but calling a class constructor to get back a different subtype is non-obvious to new users and confusing to type checkers (declared return type `Sentence`, actual return type is a subclass). A `Sentence.parse()` classmethod would improve discoverability.

### TD-004 — `load_model()` uses `pickle.load()` directly
**File**: `little_questions/models/__init__.py:66–76`
`load_model()` uses raw `pickle.load()` instead of `joblib.load()`. The rest of the codebase (e.g. `Classifier.load_from_file()` at `little_questions/classifiers/base.py:112`) uses `joblib.load()`. This inconsistency means pickle format differences could cause silent failures. `load_model()` is currently unreachable from normal usage (superseded by `Classifier.load_from_file()`), but is confusing to readers.

### TD-005 — `SentenceScorerEN` has no dataset, purely heuristic
**File**: `little_questions/classifiers/lang/en/__init__.py:22`
Comment reads "no good dataset for training, so this will work for now". The scorer is entirely rule-based and has known classification gaps (e.g. commands that start with a noun due to NLTK mistagger). Accuracy is not measured.

### TD-006 — Catalan stemmer missing
**File**: `little_questions/classifiers/lang/ca/features.py:14`
TODO comment: "better stemmer for Catalan". The pipeline falls back to English stemming for Catalan, degrading classification accuracy.

### TD-007 — French, German, Italian have no POS tagger
**File**: `little_questions/classifiers/lang/fr/`, `de/`, `it/`
Only lemmatizer + n-gram/TF-IDF features; no POS-tag vectorizer. Pipeline accuracy for these languages is likely lower than English, Spanish, and Portuguese.

### TD-008 — `MODEL2URL` URLs hardcoded to `v0.7.0a1` GitHub release tag
**File**: `little_questions/models/__init__.py:16–44`
If the release tag changes (e.g. `v0.7.0` stable), the download URLs must be manually updated. No version variable or URL template is used.

### TD-009 — `download()` has no retry or progress indicator
**File**: `little_questions/models/__init__.py:48–63`
`requests.get(url).content` blocks until complete with no streaming, progress reporting, or retry. On slow connections this silently hangs. `requests.get(url, stream=True)` with chunked write would improve UX.

### TD-010 — `pyproject.toml` missing `[project.urls]` and classifiers
**File**: `pyproject.toml`
No `Homepage`, `Repository`, `Changelog` URLs. No trove classifiers (e.g. `Natural Language :: English`, `Topic :: Scientific/Engineering :: Artificial Intelligence`). Required before PyPI publication.

---

## Security notes

- Model files are downloaded over HTTPS from GitHub releases. No checksum verification is performed. A MITM attack or compromised release could deliver a malicious pickle. Add SHA-256 verification against a hardcoded manifest.
- `pickle.load()` (TD-004) and `joblib.load()` both execute arbitrary code. Only load `.pkl` files from trusted sources.

---

## Out of scope / will not fix

- Dropping `str` subclassing for `Sentence` — intentional for voice pipeline compatibility; documented in ROADMAP.md.
- `TNaLaGmes`-era intent parsing (`docs/intents.md`) — partially deprecated; not developed further.
