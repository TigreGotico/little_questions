# Status: Language-Specific Categorical Feature Extractors

## Checklist

- [x] Create `train/lang/feature_extractors.py` with `LanguageFeatureExtractor` base class
- [x] Implement EN-specific extractor in `features_en.py` subclass
- [x] Create sklearn `LanguageFeatureTransformer` wrapper (fit/transform interface)
- [x] Modify `LinearSVCClassifier` to accept optional categorical features; build FeatureUnion pipeline
- [x] Test EN pipeline (TF-IDF + categorical) on sentence-type data; measure baseline F1
- [x] Compute MI for each EN feature; prune low-signal ones (MI < 0.01 bits)
- [x] Implement ES, CA, FR, DE, IT, NL, PT extractors with language-tuned keywords
- [x] Train and evaluate all 8 languages; measure F1 improvement vs. TF-IDF-only
- [x] Verify feature count ≤ 100 per language; fix if exceeded
- [x] Export final EN+all-langs pipelines to ONNX
- [x] Validate ONNX inference: predictions match sklearn on 100 random samples per language
- [x] Generate evaluation reports and commit all changes

## Blockers

<!-- populated if work is stuck -->
