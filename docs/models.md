# Models

All models are LinearSVC classifiers trained on TF-IDF word n-grams (1,2), exported to ONNX for inference.
Models are downloaded on first use from GitHub releases and cached in `~/.local/share/little_questions/`.

## Supported languages and model files

| Lang | 52-class model | 6-class model |
|------|----------------|---------------|
| `en` | `questions52_svm_EN_0.8.0.onnx` | `questions6_svm_EN_0.8.0.onnx` |
| `es` | `questions52_svm_ES_0.8.0.onnx` | `questions6_svm_ES_0.8.0.onnx` |
| `pt` | `questions52_svm_PT_0.8.0.onnx` | `questions6_svm_PT_0.8.0.onnx` |
| `ca` | `questions52_svm_CA_0.8.0.onnx` | `questions6_svm_CA_0.8.0.onnx` |
| `fr` | `questions52_svm_FR_0.8.0.onnx` | `questions6_svm_FR_0.8.0.onnx` |
| `de` | `questions52_svm_DE_0.8.0.onnx` | `questions6_svm_DE_0.8.0.onnx` |
| `it` | `questions52_svm_IT_0.8.0.onnx` | `questions6_svm_IT_0.8.0.onnx` |
| `nl` | `questions52_svm_NL_0.8.0.onnx` | `questions6_svm_NL_0.8.0.onnx` |

Use `"en_small"`, `"es_small"`, etc. to load the 6-class coarse-label model instead.

## Training pipeline

Feature extraction: `TfidfVectorizer(analyzer="word", ngram_range=(1,2), sublinear_tf=True)` → `LinearSVC(C=1.0)`.

The enhanced pipeline (used when `--enhanced` flag is passed to `train/compare_classifiers.py`) adds char n-grams (3,4) and 40+ linguistic features from `train/features.py` (`LinguisticFeaturesTransformer`).

## Sentence-type model

`SentenceTypeClassifier` — `little_questions/sentence_type.py`. TF-IDF word+char n-grams + LinearSVC, with a punctuation+first-word heuristic fallback (`_fallback_score`). One model file per language, loaded via `get_scorer()`.

## Download API

See [api.md](api.md) for `download()`, `get_model_path()`, and per-language helpers.

## Integrity

`MODEL2SHA256` (`little_questions/models/__init__.py:46`) maps each language code to the expected SHA-256 digest. Entries are `None` until populated after a release upload. Missing entries log a warning and skip verification. Populate this dict after uploading each release.
