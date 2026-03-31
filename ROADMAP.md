# little_questions — Roadmap

## Status Summary

| Phase | Goal | Status |
|-------|------|--------|
| 1 — Stabilize | Installable, working, tested | ✅ Done |
| 2 — Clean API | Type-safe, no magic, canonical entry point | ✅ Done |
| 3 — ONNX Migration | Replace joblib .pkl with ONNX runtime inference | ✅ Done |
| 4 — Train Models | Retrain all languages with ONNX export | ✅ Done |
| 5 — PyPI Release | Publish v0.8.0 | ⬜ Planned |

---

## Phase 1 — Stabilize ✅ Done

- [x] Replace `setup.py` with `pyproject.toml`
- [x] Fix model path bug (missing `.pkl` extensions)
- [x] Add type hints and docstrings
- [x] 40 unit tests passing

---

## Phase 2 — Clean API ✅ Done

- [x] `Sentence.parse()` canonical entry point
- [x] `clear_classifier_cache()` utility
- [x] `list_supported_languages()` utility
- [x] `SentenceScorerEN` type hints

---

## Phase 3 — ONNX Migration ✅ Done

- [x] Replace joblib-only `Classifier` with ONNX Runtime inference
- [x] `train/classifiers.py` - new trainable classifiers module
- [x] Removed JarbasModelZoo dependency (replaced with `brill_postagger`)
- [x] Added `onnxruntime` and `brill_postagger` as runtime dependencies
- [x] Removed `features.py` sklearn training code from inference package

---

## Phase 4 — Train Models ✅ Done

Trained ONNX models for 8 languages with balanced training data (4677 questions):

| Language | Accuracy | Model File |
|----------|----------|------------|
| EN | 82% | `questions52_svm_EN_0.8.0.onnx` |
| ES | 79% | `questions52_svm_ES_0.8.0.onnx` |
| CA | 79% | `questions52_svm_CA_0.8.0.onnx` |
| PT | 77% | `questions52_svm_PT_0.8.0.onnx` |
| FR | 78% | `questions52_svm_FR_0.8.0.onnx` |
| DE | 76% | `questions52_svm_DE_0.8.0.onnx` |
| IT | 78% | `questions52_svm_IT_0.8.0.onnx` |
| NL | 78% | `questions52_svm_NL_0.8.0.onnx` |

Training data: `train/clean_data/raw_questions_EN_balanced_0.8.0.txt`
Translation: argostranslate (offline)

### Sentence Type Classifier (NEW) ✅

Replaced heuristic-based `SentenceScorerEN` with trained classifier:

| Model | Accuracy | Features |
|-------|----------|----------|
| TF-IDF word(1,2)+char(3,4) | 86.3% | lexical |
| M2V multilingual-128M | 91.2% | embeddings |
| **TF-IDF + M2V combo** | **93.0%** | combined |

Models: `little_questions/models/sentence_type_svm_EN.joblib`
Training data: `train/clean_data/sentence_types_EN.txt` (628 sentences)
Labels: question, statement, command, exclamation, request

---

## Phase 5 — PyPI Release 🔄 In Progress

- [ ] Upload ONNX models to GitHub releases (tag `0.8.0`)
- [ ] Update CHANGELOG.md
- [ ] `pip install little-questions[train]` for training
- [ ] Publish to PyPI

---

## Supported Languages

| Language | Code | POS Tagger | COSC Model |
|----------|------|------------|------------|
| English | en | NLTK (built-in) | ✅ 82% |
| Spanish | es | brill_postagger | ✅ 79% |
| Portuguese | pt | brill_postagger | ✅ 77% |
| Catalan | ca | brill_postagger | ✅ 79% |
| French | fr | brill_postagger | ✅ 78% |
| German | de | brill_postagger | ✅ 76% |
| Italian | it | brill_postagger | ✅ 78% |
| Dutch | nl | brill_postagger | ✅ 78% |

---

## Architecture

```
little_questions/          # Inference package (ONNX Runtime)
├── Sentence.parse()      # Main entry point
├── Sentence(text)       # Classify sentences
└── ONNX Runtime          # Model inference

train/                    # Training package (sklearn)
├── classifiers.py        # Trainable classifiers
├── train_all.py          # Train all languages
├── translate.py         # Translate training data
└── clean_data/          # Training datasets
```
