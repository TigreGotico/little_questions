# MAINTENANCE_REPORT

## 2026-03-31 — Production Readiness Audit (v0.8.0)

**AI Model**: claude-sonnet-4-6
**Actions Taken**:

1. **Fixed `pyproject.toml`** (`pyproject.toml`):
   - Added missing `requests` runtime dependency (used in `little_questions/models/__init__.py:download`).
   - Added trove classifiers (`Development Status`, `Intended Audience`, `Natural Language`,
     `Programming Language`, `Topic`).
   - Added `Repository`, `Changelog`, and `Bug Tracker` project URLs for PyPI.

2. **Wrote complete CHANGELOG.md** (`CHANGELOG.md`):
   - Added v0.8.0 entry documenting ONNX migration, new API, test suite, and removed items.
   - Added v0.7.0a1 entry for historical record.
   - Added comparison URLs for all tagged versions.

3. **Updated AUDIT.md** (`AUDIT.md`):
   - Moved BUG-001, TD-002, TD-003, TD-004, TD-005, TD-008, TD-010 to "Resolved (v0.8.0)" table.
   - Updated file references from removed `features.py` to current `postag.py` / `legacy.py`.
   - Retained open items: TD-001, TD-006, TD-007, TD-009 (non-blocking).

4. **Updated FAQ.md** (`FAQ.md`):
   - Corrected `.pkl` → `.onnx` model format reference.
   - Added `nl` (Dutch) to supported language list (was missing).
   - Updated classifier/training section to reflect `train/` package and ONNX format.
   - Updated sentence-type section to reference `SentenceTypeClassifier` (trained, 93%).
   - Updated "add a new language" instructions to match v0.8 structure.

5. **Updated SUGGESTIONS.md** (`SUGGESTIONS.md`):
   - Moved S-001, S-002, S-006, S-007 to "Completed" table.
   - Retained open items: S-003, S-004, S-005, S-008.

**Oversight**: All changes reviewed by human before commit.
**Test result**: 40/40 passed.

---

## 2026-03-31 — Code-smell refactor

**AI Model**: claude-sonnet-4-6
**Actions Taken**:

1. **Fixed logic bugs in `legacy.py`** (S-009, S-010, S-017):
   - `exclamation_score`: `not A or not B` (always True) → `not (A or B)`
   - Five `not x in y` occurrences replaced with `x not in y`
   - `YES_NO_STARTERS` list had duplicate `"has"`; converted to `frozenset`

2. **Replaced `pretty_label` if/elif chain with dict lookups** (S-014):
   - Added `_MAIN_LABEL_NAMES` and `_SEC_LABEL_NAMES` dicts at module level in `__init__.py`
   - Reduced 50+ lines to 2 dict lookups

3. **Removed dead HTTP scorer branch** (S-020):
   - `Sentence.__new__` had an unresolvable TODO for HTTP URL lang extraction
   - `get_model_path()` raises `NotImplementedError` for HTTP anyway; branch deleted

4. **Eliminated class/function duplication** (S-011, S-012, S-016):
   - `SentenceScorer` consolidated to `base.py`; duplicate in `classifiers/__init__.py` removed
   - Dead `Classifier` class in `base.py` removed (only `classifiers/__init__.py` version was used)
   - `get_scorer()` was defined in three places; now only in `classifiers/__init__.py`
   - `SENTENCE_TYPES` and `SUPPORTED_LANGUAGES` extracted to `little_questions/constants.py`

5. **Removed dead lang sub-packages** (S-015):
   - Deleted `classifiers/lang/{ca,de,es,fr,it,nl,pt}/` (11 files, all unused by inference)
   - Moved `postag.py` files (ca, es, pt) + `pt/tokenize.py` to `train/lang/`
   - Fixed import in `train/lang/pt/postag.py`

6. **Extracted shared `load_data()` to `train/utils.py`** (S-021):
   - Duplicate `load_data()` in `train_en.py` and `train_all.py` consolidated
   - `train_all.py` version (which also skipped blank lines) used as canonical

7. **Added `.gitignore`**: excludes `__pycache__`, `*.pyc`, `*.joblib`, `*.pkl`, `*.npy`, `*.onnx`

**Oversight**: All changes reviewed by human before commit.
**Test result**: 40/40 passed after each commit.

---

## 2026-03-30 — Phase 1 Revival

**AI Model**: claude-sonnet-4-6
**Actions Taken**:

1. **Replaced fragile `__getattribute__` delegation** in `Sentence` (`little_questions/__init__.py`).
   - Removed the `__getattribute__` override that intercepted all `str` method calls and
     attempted to re-wrap return values in the same `Sentence` subclass by calling `__new__`.
     This was fragile because it called the full classifier pipeline for every string
     operation (`.upper()`, `.split()`, etc.), causing infinite recursion and unexpected
     classification side-effects.
   - Replaced with a clean `str.__new__(target_cls, content)` call and a minimal `__getattr__`
     safety net that only fires when normal attribute lookup fails.
   - The correct concrete subclass (`Question`, `Command`, etc.) is now selected via a
     plain dict lookup (`_TYPE_MAP`) rather than a chain of `super().__new__()` calls.

2. **Added type hints and docstrings** to:
   - `Sentence`, `Question`, `Command`, `Request`, `Exclamation`, `Statement`
     (`little_questions/__init__.py`)
   - `SentenceScorer`, `Classifier` and all `*TextClassifier` subclasses
     (`little_questions/classifiers/base.py`)

3. **Wrote unit tests** (`test/test_sentence.py`):
   - 28 tests covering: correct subclass dispatch, all `is_*` boolean properties,
     all 6 COSC main labels, `pretty_label` formatting, str behaviour, and
     `SentenceScorer` heuristics.
   - All tests run without network access or model files (classifiers are mocked).
   - `test/conftest.py` stubs `JarbasModelZoo` and `xdg`/`pyxdg` for CI environments
     where those packages are not installed.

4. **Created `docs/index.md`** — overview, COSC taxonomy table, sentence type table,
   usage example, and architecture diagram.

**Oversight**: All changes reviewed by human before commit.
**Test result**: 28/28 passed.
