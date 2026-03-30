# little_questions — Roadmap

## Status summary

| Phase | Goal | Status |
|-------|------|--------|
| 1 — Stabilize | Installable, working, tested | ✅ Done (2026-03-30) |
| 2 — Clean API | Type-safe, no magic, canonical entry point | 🔄 In progress |
| 3 — OVOS integration | Pipeline adapter, voice assistant routing | ⬜ Planned |
| 4 — PyPI release | Published as `little-questions` | ⬜ Planned |

---

## Phase 1 — Stabilize ✅ Done

- [x] Replace `setup.py` with `pyproject.toml`; `python_requires = ">=3.10"`
- [x] Replace fragile `__getattribute__` delegation in `Sentence` with `__new__` + `_TYPE_MAP`
- [x] Add type hints and docstrings to `Sentence`, `Classifier`, `SentenceScorer` and all subclasses
- [x] Write 28 unit tests (sentence type dispatch, COSC labels, str behaviour, scorer heuristics)
- [x] `test/conftest.py` stubs `JarbasModelZoo` and `xdg` so tests run without network/models
- [x] `docs/index.md` — overview, COSC taxonomy, sentence type table, usage example, architecture

---

## Phase 2 — Clean API 🔄 In progress

**Goal:** Ergonomic, typed, documented entry points. No implicit magic at import time.

### 2a — Fix model path bug
- [ ] `LANG2MODEL` entries for `ca`, `fr`, `it`, `de` are missing `.pkl` extension
  — `get_model_path()` returns paths that `joblib.load()` cannot open
  — Fix: append `.pkl` to those four entries in `little_questions/models/__init__.py`
  — Add regression test: `get_model_path(lang)` returns a string ending in `.pkl` for all 7 languages

### 2b — Add `Sentence.parse()` canonical entry point
- [ ] Add `Sentence.parse(text: str, lang: str = "en") -> "Sentence"` classmethod
  — mirrors `Sentence(text, model=lang)` but has a discoverable, typed signature
  — docstring explains that the returned type is a concrete subclass
  — keep `Sentence(text, model=lang)` working unchanged (no breaking change)

### 2c — Simplify and document `get_classifier()`
- [ ] Add full docstring to `get_classifier()` explaining the lazy cache (`_LAZY_LOADING`)
- [ ] Add `clear_classifier_cache()` utility for test isolation and memory management
- [ ] Add `list_supported_languages() -> List[str]` returning the 7 supported language codes

### 2d — SentenceScorerEN type hints
- [ ] Add return type annotations to all `SentenceScorerEN` static methods
  — `predict(text: str) -> str`
  — `score(text: str) -> Dict[str, float]`
  — `_score(...) -> float`
  — `*_score(text: str) -> float`

### 2e — Expand test coverage
- [ ] Test `get_model_path()` returns `.pkl` paths for all 7 languages (no network needed)
- [ ] Test `Sentence.parse()` classmethod produces the same result as `Sentence()` constructor
- [ ] Test `clear_classifier_cache()` resets `_LAZY_LOADING`
- [ ] Test `list_supported_languages()` returns all 7 codes

---

## Phase 3 — OVOS integration ⬜ Planned

**Goal:** Drop-in adapter for routing utterances in an OVOS skill pipeline.

- [ ] `little_questions.ovos.QuestionRouter` — thin wrapper returning an intent-compatible dict:
  ```python
  {"utterance": text, "main_label": "HUM", "secondary_label": "ind",
   "sentence_type": "question", "confidence": 0.82}
  ```
- [ ] Register as an OVOS utterance transformer plugin (entry point in `pyproject.toml`)
- [ ] CI: add Python 3.10 / 3.11 / 3.12 GitHub Actions workflow
- [ ] Smoke test per language (requires pre-downloaded model fixtures in CI or mocked)

---

## Phase 4 — PyPI release ⬜ Planned

**Goal:** Publish as `little-questions` so downstream projects can `uv add little-questions`.

- [ ] Retrain all 7 language models on Python 3.12 + scikit-learn ≥ 1.4 to eliminate joblib pickle version mismatch warnings
- [ ] Bundle small NLTK data assets (`punkt`, `averaged_perceptron_tagger`) as package data or download on first use with a friendly message
- [ ] `CHANGELOG.md` — add v0.7.0 release entry covering all Phase 1–3 changes
- [ ] `pyproject.toml` classifiers, keywords, project URLs
- [ ] Build and publish to PyPI

---

## Key invariants

The COSC 52-class taxonomy (`main_label` / `secondary_label` / `pretty_label`) is the primary
deliverable. Any refactor must preserve the full label set and be covered by regression tests
before the classifier pipeline is touched.

`Sentence` subclassing `str` is intentional (voice assistant pipelines treat utterances as
strings; subclassing lets metadata travel with the string without changing call sites). If this
ever causes real problems, introduce `Sentence.parse()` as the replacement entry point first,
then deprecate the `Sentence(str)` constructor in a separate breaking-change release.
