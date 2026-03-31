# SUGGESTIONS — little_questions

Agent-generated improvement proposals. Each item is scoped, actionable, and non-breaking.

---

## Open

### S-008 — Publish sentence-type accuracy per language

**Priority**: Medium
**Effort**: Medium

`train/eval_sentence_type.py` and the evaluation harness in `train/metrics.py` now
exist. The missing step is: create labelled `sentence_types_*.txt` datasets for all
supported languages (currently only EN exists), run the evaluation, and publish results
in `docs/classification.md`. Without data, non-English sentence-type accuracy is
unknown.

---

### S-013 — Replace `SentenceScorer` static-only class with module-level functions

**Priority**: Low (design smell)
**Effort**: Small

`train/baselines.py:HeuristicScorer` is a collection of `@staticmethod` methods with
no instance state — namespace misuse. Convert to module-level functions. Public API
surface shrinks and the code becomes easier to import and test.

---

## Completed (post v0.8.0 refactor — 2026-03-31)

| ID | Suggestion | Resolved |
|----|-----------|---------|
| S-003 | Stream model downloads with progress | `little_questions/models/__init__.py:_download_file` — chunked streaming, 3 retries |
| S-004 | Add SHA-256 checksum verification for downloaded models | `little_questions/models/__init__.py:MODEL2SHA256`, `_verify_checksum` |
| S-005 | Add `classify()` fast path | `little_questions/__init__.py:classify`, `classify_batch` |
| S-009 | Fix `or` logic bug in `exclamation_score` | Fixed in `train/baselines.py:HeuristicScorer` before moving |
| S-010 | Fix `not x in y` anti-pattern | Fixed in `train/baselines.py:HeuristicScorer` before moving |
| S-011 | Eliminate `SentenceScorer`/`Classifier` duplication | `classifiers/base.py` and `classifiers/legacy.py` deleted; moved to `train/baselines.py` |
| S-012 | Remove `get_scorer()` triplicate | Single definition in `classifiers/__init__.py` |
| S-014 | Replace `pretty_label` if/elif chain | `__init__.py:_MAIN_LABEL_NAMES`, `_SEC_LABEL_NAMES` |
| S-015 | Remove dead lang sub-packages | Deleted `classifiers/lang/{ca,de,es,fr,it,nl,pt}/`; postag files moved to `train/lang/` |
| S-016 | Consolidate duplicate constants | `little_questions/constants.py` |
| S-017 | Fix duplicate `"has"` in `YES_NO_STARTERS` | Converted to `frozenset` in `train/baselines.py` |
| S-018 | Tighten `Optional[Any]` type hints | `TYPE_CHECKING` guards in `sentence_type.py`, `classifiers/__init__.py` |
| S-019 | Thread-safe classifier caches | `threading.Lock` in `classifiers/__init__.py` and `sentence_type.py` |
| S-020 | Remove dead HTTP TODO branch | Removed from `Sentence.__new__` |
| S-021 | Extract `load_data()` to `train/utils.py` | `train/utils.py:load_data` |

## Completed (v0.8.0)

| ID | Suggestion | Resolved |
|----|-----------|---------|
| S-001 | Add `Sentence.parse()` classmethod | `little_questions/__init__.py:Sentence.parse` |
| S-002 | Add `clear_classifier_cache()` and `list_supported_languages()` | `little_questions/classifiers/__init__.py` |
| S-006 | Replace `pickle.load()` with `joblib.load()` in `load_model()` | `load_model()` removed; `Classifier.load_from_file()` uses ONNX/joblib |
| S-007 | Add trove classifiers and project URLs to `pyproject.toml` | Done in v0.8.0 |
