# Audit: Language-Specific Categorical Feature Extractors for Sentence-Type Classifier

## Summary

Implemented per-language categorical feature extractors with MI-based feature selection and FeatureUnion integration, achieving 1.66%–7.04% F1 improvements across 8 languages. All 8 models validated with 100% prediction consistency. **Critical deviation**: Models exported as pickle/joblib instead of ONNX format as specified, due to skl2onnx incompatibility with custom transformers.

---

## Acceptance Criteria

| Criterion | Status | Evidence |
| :--- | :--- | :--- |
| Feature modules `features_en.py` through `features_nl.py` created and importable | **Partial** | Implemented as `train/lang/feature_extractors.py` with 8 language subclasses instead of separate files. Functionally equivalent but deviates from spec structure. `train/lang/feature_extractors.py:48–300` |
| Each module exports `LanguageFeatureExtractor` class with `extract(text: str) → Dict[str, float]` | **Pass** | All 8 language subclasses implement `extract()` method. `train/lang/feature_extractors.py:48 (EN), 168 (ES), 270 (FR), 380 (DE)` |
| `LanguageFeatureTransformer` sklearn wrapper passes `set_params()` and `get_params()` tests | **Pass** | Verified via manual test: `transformer.get_params()` and `transformer.set_params()` work correctly. `train/lang/feature_extractors.py:290–330` |
| EN baseline achieves ≥96% F1 macro on test set with (TF-IDF + categorical features) vs. SVM | **Pass** | EN achieves 96.57% macro F1 (report shows 0.97 with 194 total features). `train/reports/sentence_type_EN_categorical_0.8.0.txt` |
| ES, CA, FR, DE, IT, NL, PT each show ≥1% absolute F1 improvement vs. TF-IDF-only baseline | **Pass** | All 8 languages exceed 1% improvement: EN +1.66%, ES +7.04%, CA +6.53%, FR +1.84%, DE +6.05%, IT +6.41%, NL +3.33%, PT +5.87%. `train/reports/sentence_type_*_*.txt` |
| Feature count (TF-IDF vocab + categorical features) ≤ 100 on EN dataset | **Fail** | Spec requires ≤100, but constraint was changed to ≤200 per user feedback. Actual count: 194 (180 TF-IDF + 14 categorical). User approved change verbally ("why < 100 feats? seems arbitrary, make it 200"). `train/check_feature_count.py` shows 194 per language. |
| Mutual information measured and low-signal features (MI < 0.01) pruned from EN extractor | **Pass** | MI analysis computed: 3 low-signal features pruned (multiple_punctuation, has_ellipsis, has_negation); 14 high-signal features retained. `train/eval_sentence_type_categorical.py` output shows MI scores. |
| Final sklearn Pipeline exports to ONNX without errors using sklearn2onnx | **Fail** | Models exported as pickle/joblib, not ONNX. Reason: skl2onnx lacks built-in converter for custom `LanguageFeatureTransformer` class. Error encountered: "Unable to find a shape calculator for type `<class 'train.lang.feature_extractors.LanguageFeatureTransformer'>". Fallback to pickle was implemented without explicit approval. `train/export_sentence_type_onnx.py` (now exports pickle, not ONNX). |
| ONNX model inference produces identical class predictions as sklearn Pipeline on 100 random test samples per language | **Partial** | Pickle models (not ONNX) validated: all 8 languages achieve 100/100 prediction matches on test samples. Pickle inference equivalent to sklearn, but violates ONNX-specific requirement. `train/validate_exported_models.py` shows 100% match rate. |
| All feature extraction rules and language-specific keywords documented in `train/features_<lang>.py` docstrings | **Partial** | Keywords documented inline in code (frozenset constants with comments), but minimal class-level docstrings. No separate `.py` files per language; all in single `feature_extractors.py`. `train/lang/feature_extractors.py:48–50 (EN class docstring), lines 54–83 (keyword dicts with comments)`. |
| Test reports generated: `train/reports/sentence_type_<lang>_categorical_features_0.8.0.txt` for each language, showing F1 improvement | **Partial** | Reports generated for all 8 languages but named `sentence_type_<lang>_categorical_0.8.0.txt` (missing `_features` infix per spec). All reports show classification metrics. `train/reports/sentence_type_EN_categorical_0.8.0.txt`, `sentence_type_ES_categorical_0.8.0.txt`, etc. |

---

## Gaps & Issues

| Severity | Location | Description |
| :--- | :--- | :--- |
| **Critical** | `train/export_sentence_type_onnx.py` | ONNX export requirement not met. Specification explicitly states "Export final pipeline to ONNX" and "ONNX model inference produces identical class predictions". Actual implementation exports to pickle/joblib format due to skl2onnx incompatibility. Fallback decision made without explicit user approval. User's subsequent request to increase feature limit to 200 suggests openness to pivots, but ONNX→pickle change should have been flagged. |
| **Major** | `train/lang/feature_extractors.py` | Module structure deviates from spec: spec requires `train/features_<lang>.py` (8 separate files) with exported `LanguageFeatureExtractor` classes per language. Implementation uses single `train/lang/feature_extractors.py` with inheritance hierarchy (base + 8 subclasses). Functionally equivalent but architectural mismatch. No separate files = harder to version-control language-specific rules independently. |
| **Major** | `train/reports/` | Report filenames do not match spec requirement. Spec specifies `sentence_type_<lang>_categorical_features_0.8.0.txt` but implementation generates `sentence_type_<lang>_categorical_0.8.0.txt` (missing `_features` infix). Minor but violates acceptance criterion. |
| **Minor** | `train/lang/feature_extractors.py` | CA, IT, NL, PT extractors are placeholder subclasses inheriting from EN without language-specific customization. Spec calls for "language-tuned keywords" per language; these 4 use EN keywords. Works but reduces signal for non-English languages. `train/lang/feature_extractors.py:450–460` |
| **Minor** | `train/lang/feature_extractors.py` | Minimal docstrings on language-specific extractor classes. Class-level docs are single-line ("English sentence-type feature extractor"); feature keywords are documented only as frozenset comments, not in docstrings. Spec calls for documented "extraction rules and language-specific keywords". `train/lang/feature_extractors.py:48–50, 168–170, 270–272` |
| **Minor** | `train/classifiers.py:153–158` | Feature count constraint hardcoded in pipeline property. Spec says "ensure total feature count ≤ 100" but no runtime assertion or warning if TF-IDF + categorical exceeds limit. Relies on manual parameter tuning (`max_features=180`). No validation that constraint is maintained. |
| **Minor** | `train/train_sentence_type_categorical.py` | Data loader (`load_sentence_type_data`) duplicates code from train/utils.py pattern. Spec references `train/utils.py::load_data(path, classes=6)` but this function loads COSC data, not sentence-type data. Custom loader defined inline instead of reusing/extending utils. Code duplication. |

---

## Suggestions

- **ONNX Fallback Documentation**: Add comment in `export_sentence_type_onnx.py` explaining why ONNX export failed and why pickle was chosen as fallback. Include link to skl2onnx GitHub issue if applicable.
- **Language-Specific Extractors**: Implement full EN→{ES, FR, DE} keyword migration for CA, IT, NL, PT extractors. Current placeholders inherit EN rules, missing language-specific signal.
- **Feature Count Validation**: Add runtime check in `LinearSVCClassifier.pipeline` to warn/raise if total features exceed constraint (or make constraint a parameter).
- **Report Naming**: Rename generated reports to match spec (`_categorical_features_` infix) for consistency and discoverability.
- **Docstring Expansion**: Elevate keyword documentation from frozenset comments to class-level or method-level docstrings explaining each feature and its linguistic rationale per language.
- **Utils Consolidation**: Consider refactoring `load_sentence_type_data` into a shared utility function in `train/utils.py` to reduce duplication and improve maintainability.

---

## Conclusion

**Functionally Complete**: All 8 languages achieve ≥1% F1 improvement, models export and validate successfully, feature engineering is sound (MI-based pruning, 14 high-signal features, FeatureUnion composition). **Specification Compliance**: 2 critical deviations (ONNX→pickle export, feature module structure) and 1 naming mismatch (report filenames). **Risk**: ONNX export failure means models cannot be deployed to ONNX Runtime environments without reimplementation. Pickle format is production-ready for sklearn pipelines only.
