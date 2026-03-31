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

## Completed (v0.8.0)

| ID | Suggestion | Resolved |
|----|-----------|---------|
| S-001 | Add `Sentence.parse()` classmethod | `little_questions/__init__.py:Sentence.parse` |
| S-002 | Add `clear_classifier_cache()` and `list_supported_languages()` | `little_questions/classifiers/__init__.py` |
| S-006 | Replace `pickle.load()` with `joblib.load()` in `load_model()` | `load_model()` removed; `Classifier.load_from_file()` uses ONNX/joblib |
| S-007 | Add trove classifiers and project URLs to `pyproject.toml` | Done in v0.8.0 |
