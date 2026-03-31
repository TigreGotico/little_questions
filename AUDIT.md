# AUDIT — little_questions

Known issues, technical debt, and security notes. Updated 2026-03-31.

---

## Critical bugs

*None open. See resolved section below.*

---

## Technical debt

### TD-001 — `SentenceScorerHeuristic` lacks type hints
**File**: `little_questions/classifiers/legacy.py`
**Severity**: Low — internal fallback, not part of public API

All static methods in `SentenceScorerHeuristic` have no type annotations. Inconsistent
with the rest of the codebase post-Phase 2 refactor.

### TD-006 — Catalan stemmer missing
**File**: `little_questions/classifiers/lang/ca/postag.py`
**Severity**: Low — documented limitation

Falls back to English `PorterStemmer` for Catalan, degrading COSC accuracy from ~82%
(EN) to ~79% (CA). A dedicated Catalan stemmer would improve accuracy.

### TD-007 — FR/DE/IT/NL have no POS tagger
**File**: `little_questions/classifiers/lang/fr/`, `de/`, `it/`
**Severity**: Low — n-gram features still produce 76–78% accuracy

Only TF-IDF features used; no POS-tag vectorizer. `brill_postagger` has models for
these languages; adding them would improve COSC accuracy.

### TD-009 — `download()` has no retry or progress indicator
**File**: `little_questions/models/__init__.py:76`
**Severity**: Low — UX only, not a correctness issue

`requests.get(url, timeout=300).content` blocks until complete with no streaming or
progress. A chunked download with retry would improve UX on slow connections.

---

## Security notes

- **No checksum verification**: ONNX model files are downloaded over HTTPS from
  GitHub releases but SHA-256 checksums are not verified. A compromised release or
  MITM could deliver a malicious model. Mitigation: add a hardcoded SHA-256 manifest
  and verify after download.
- **`joblib.load()` executes arbitrary code** (`.pkl` fallback path). Only load from
  trusted sources; ONNX models are preferred and do not have this risk.

---

## Out of scope / will not fix

- Dropping `str` subclassing for `Sentence` — intentional for voice pipeline
  compatibility; documented in ROADMAP.md.
- `TNaLaGmes`-era intent parsing (`docs/intents.md`) — partially deprecated; not
  developed further.

---

## Resolved (v0.8.0)

| ID | Issue | Resolution |
|----|-------|------------|
| BUG-001 | Missing `.onnx` extension in `LANG2MODEL` for ca/fr/it/de | Fixed: all entries now use `.onnx` format |
| TD-002 | `get_classifier()` undocumented cache | Fixed: `clear_classifier_cache()` added |
| TD-003 | `Sentence` constructor-only entry point | Fixed: `Sentence.parse()` classmethod added |
| TD-004 | `load_model()` used `pickle.load()` directly | Fixed: `load_model()` removed; `Classifier.load_from_file()` uses joblib/ONNX |
| TD-005 | `SentenceScorerEN` purely heuristic | Fixed: replaced with trained `SentenceTypeClassifier` (93% EN accuracy) |
| TD-008 | `MODEL2URL` URLs hardcoded to v0.7.0a1 | Fixed: URLs now point to `0.8.0` release tag |
| TD-010 | `pyproject.toml` missing classifiers/URLs | Fixed: trove classifiers, URLs, and `requests` dep added |
