# MAINTENANCE_REPORT

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
