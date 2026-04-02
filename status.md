# Status: Language-Specific Categorical Feature Extractors

## Checklist

- [x] Create `train/lang/feature_extractors.py` with `LanguageFeatureExtractor` base class
- [x] Implement EN-specific extractor in `features_en.py` subclass
- [x] Create sklearn `LanguageFeatureTransformer` wrapper (fit/transform interface)
- [ ] Modify `LinearSVCClassifier` to accept optional categorical features; build FeatureUnion pipeline
- [ ] Test EN pipeline (TF-IDF + categorical) on sentence-type data; measure baseline F1
- [ ] Compute MI for each EN feature; prune low-signal ones (MI < 0.01 bits)
- [ ] Implement ES, CA, FR, DE, IT, NL, PT extractors with language-tuned keywords
- [ ] Train and evaluate all 8 languages; measure F1 improvement vs. TF-IDF-only
- [ ] Verify feature count ≤ 100 per language; fix if exceeded
- [ ] Export final EN+all-langs pipelines to ONNX
- [ ] Validate ONNX inference: predictions match sklearn on 100 random samples per language
- [ ] Generate evaluation reports and commit all changes

## Blockers

<!-- populated if work is stuck -->
