# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.0] - 2026-03-31

### Added
- ONNX Runtime inference for COSC question classifiers (16 models: 52-class and 6-class per language)
- `brill_postagger` dependency replaces `JarbasModelZoo` for POS tagging in ES/PT/CA
- Trained sentence type classifier for English (TF-IDF + M2V combo, 93% accuracy)
- `Sentence.parse(text, lang)` classmethod as canonical entry point
- `clear_classifier_cache()` and `list_supported_languages()` utilities
- `train/` package: `TrainableClassifier`, `LinearSVCClassifier`, `LogRegTextClassifier`,
  `NaiveBayesTextClassifier` pipelines for retraining
- `train/translate_dataset.py` for multilingual dataset generation via argostranslate
- 40 unit tests, all passing; mocked classifiers enable CI without model files
- Type hints and docstrings on all public API

### Changed
- Replaced `setup.py` with `pyproject.toml` (PEP 517/518)
- `Classifier` now loads ONNX models via `onnxruntime`; legacy `.pkl` joblib still supported as fallback
- Moved sklearn feature extraction (`features.py`) to `train/classifiers.py` — inference package no longer depends on training infrastructure
- `download()` fetches from GitHub releases tag `0.8.0`
- `pyproject.toml` now lists `requests`, trove classifiers, and project URLs

### Fixed
- Missing `.onnx` extension in `LANG2MODEL` for ca/fr/it/de (was `.pkl` path without extension)
- Fragile `__getattribute__` delegation in `Sentence` replaced with clean `str.__new__` + `_TYPE_MAP` dispatch

### Removed
- `JarbasModelZoo` dependency
- `features.py` training code from inference package (moved to `train/`)
- Raw training data files and reports from `train_scripts/` (superseded by `train/`)

## [0.7.0a1] - 2021

### Added
- Initial multilingual support: ES, PT, CA, FR, DE, IT (via Google Translate dataset)
- `SentenceScorerEN` POS-tag heuristic for sentence type detection
- joblib `.pkl` models for 8 languages (6-class and 52-class COSC)

### Changed
- Migrated model caching to XDG Base Directory (`~/.local/share/little_questions/`)

## [0.5.2] - 2019-12-12

### Changed
- Transferred ownership to [OpenJarbas](https://github.com/OpenJarbas)

### Fixed
- Updated deprecated `simple_NER` packages

[unreleased]: https://github.com/OpenJarbas/little_questions/compare/0.8.0...HEAD
[0.8.0]: https://github.com/OpenJarbas/little_questions/compare/0.7.0a1...0.8.0
[0.7.0a1]: https://github.com/OpenJarbas/little_questions/compare/0.5.2...0.7.0a1
[0.5.2]: https://github.com/OpenJarbas/little_questions/tree/0.5.2
