# AUDIT — little_questions

Known issues, technical debt, and security notes. Updated 2026-03-31.

---

## Critical bugs

*None open.*

---

## Technical debt

### TD-006 — Catalan stemmer missing
**File**: `train/lang/ca_postag.py`
**Severity**: Low — documented limitation

Falls back to English `PorterStemmer` for Catalan, degrading COSC accuracy from ~82%
(EN) to ~79% (CA). A dedicated Catalan stemmer would improve accuracy.

### TD-007 — FR/DE/IT/NL have no POS tagger
**File**: `train/train_all.py` (feature extraction via `LinearSVCClassifier`)
**Severity**: Low — n-gram features still produce 76–78% accuracy

Only TF-IDF features used for these languages; no POS-tag vectorizer. `brill_postagger`
has models for FR/DE/IT; adding them would improve COSC accuracy.

### TD-011 — `MODEL2SHA256` checksums are unpopulated
**File**: `little_questions/models/__init__.py:MODEL2SHA256`
**Severity**: Medium (security) — verification code exists but is a no-op

All entries are `None`, so SHA-256 verification is silently skipped on every download.
Checksums must be computed from the published release assets and hardcoded here.
Until then, corrupted or tampered downloads are not caught.

---

## Security notes

- **`joblib.load()` executes arbitrary code** (`.pkl` fallback path in
  `little_questions/classifiers/__init__.py`). Only load from trusted sources;
  ONNX models are preferred and do not have this risk.
- **SHA-256 checksums not yet populated** — see TD-011.

---

## Out of scope / will not fix

- Dropping `str` subclassing for `Sentence` — intentional for voice pipeline
  compatibility; documented in `ROADMAP.md`.
- `TNaLaGmes`-era intent parsing — partially deprecated; not developed further.

---

## Resolved

| ID | Issue | Resolution |
|----|-------|------------|
| BUG-001 | Missing `.onnx` extension in `LANG2MODEL` for ca/fr/it/de | All entries now use `.onnx` |
| TD-001 | `SentenceScorerHeuristic` lacks type hints | Class moved to `train/baselines.py:HeuristicScorer`; type hints added |
| TD-002 | `get_classifier()` undocumented cache | `clear_classifier_cache()` added |
| TD-003 | `Sentence` constructor-only entry point | `Sentence.parse()` classmethod added |
| TD-004 | `load_model()` used `pickle.load()` directly | Removed; `Classifier.load_from_file()` uses joblib/ONNX |
| TD-005 | `SentenceScorerEN` purely heuristic | Replaced with trained `SentenceTypeClassifier` |
| TD-008 | `MODEL2URL` URLs hardcoded to v0.7.0a1 | URLs point to `0.8.0` release tag |
| TD-009 | `download()` had no retry or progress | `_download_file()` — chunked streaming, 3 retries, 2 s backoff |
| TD-010 | `pyproject.toml` missing classifiers/URLs | Trove classifiers, URLs, `requests` dep added |
| TD-012 | No labelled sentence-type datasets for non-EN languages | Resolved: translated datasets present for all 8 languages (3001 samples each) in `train/clean_data/` |
