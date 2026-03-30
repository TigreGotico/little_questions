# SUGGESTIONS — little_questions

Agent-generated improvement proposals. Each item is scoped, actionable, and non-breaking.

---

## S-001 — Add `Sentence.parse()` classmethod

**Priority**: High
**Effort**: Small (< 30 lines)

A typed, discoverable entry point:
```python
@classmethod
def parse(cls, text: str, lang: str = "en") -> "Sentence":
    """Classify *text* and return the appropriate Sentence subclass."""
    return cls(text, model=lang)
```
Benefits: type checkers can infer the return type from `cls`, IDE autocomplete works, new users find it via `Sentence.parse(` rather than needing to know the constructor signature. See ROADMAP.md Phase 2b.

---

## S-002 — Add `clear_classifier_cache()` and `list_supported_languages()`

**Priority**: Medium
**Effort**: Small (< 20 lines)

```python
def clear_classifier_cache() -> None:
    """Clear the lazy-load classifier cache (frees RAM; models reload on next use)."""
    global _LAZY_LOADING
    _LAZY_LOADING.clear()

def list_supported_languages() -> List[str]:
    """Return the list of supported language codes."""
    return ["en", "es", "pt", "ca", "fr", "de", "it"]
```
Export both from `little_questions/__init__.py`. Useful for tests that need cache isolation and for downstream code that wants to enumerate supported models.

---

## S-003 — Stream model downloads with progress

**Priority**: Low
**Effort**: Small

Replace the blocking `requests.get(url).content` in `download()` with a streaming download:
```python
with requests.get(url, stream=True) as r:
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
```
Also add a `timeout` parameter and at least one retry on transient failures.

---

## S-004 — Add SHA-256 checksum verification for downloaded models

**Priority**: Medium (security)
**Effort**: Small

Add a `MODEL2SHA256` dict mapping model ID to expected hex digest. After download, verify with `hashlib.sha256`. Refuse to load if mismatch. Protects against corrupted downloads and MITM attacks on the model files.

---

## S-005 — Add `questions_only` fast path

**Priority**: Low
**Effort**: Small

When `Sentence` is used purely for COSC classification (not sentence-type detection), instantiating `SentenceScorerEN` (which imports NLTK and tokenises the input) wastes ~10 ms. A `Question(text, model="en", skip_scorer=True)` fast path or a `classify(text, lang)` function that returns only `(main_label, secondary_label, score)` would benefit high-throughput pipelines.

---

## S-006 — Replace `pickle.load()` with `joblib.load()` in `load_model()`

**Priority**: Low
**Effort**: Trivial

`little_questions/models/__init__.py:load_model()` uses `pickle.load()` directly. Replacing with `joblib.load()` is consistent with `Classifier.load_from_file()` and avoids any future protocol divergence.

---

## S-007 — Add trove classifiers and project URLs to `pyproject.toml`

**Priority**: Low (needed before PyPI)
**Effort**: Trivial

```toml
[project.urls]
Homepage = "https://github.com/OpenJarbas/little_questions"
Repository = "https://github.com/OpenJarbas/little_questions"
Changelog = "https://github.com/OpenJarbas/little_questions/blob/master/CHANGELOG.md"

[project.classifiers]
"Natural Language Processing"
"Topic :: Scientific/Engineering :: Artificial Intelligence"
"License :: OSI Approved :: MIT License"
"Programming Language :: Python :: 3.10"
"Programming Language :: Python :: 3.11"
"Programming Language :: Python :: 3.12"
```

---

## S-008 — Measure and publish sentence-type accuracy

**Priority**: Medium
**Effort**: Medium

`SentenceScorerEN` is entirely heuristic. Create a small hand-labelled evaluation set (100–200 sentences covering all 5 types) and add a `scripts/eval_scorer.py` that prints a confusion matrix. Publish the accuracy figure in `docs/classification.md`. This lets users make informed decisions about when to rely on sentence-type detection.
