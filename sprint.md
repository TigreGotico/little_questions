# Sprint: Language-Specific Categorical Feature Extractors for Sentence-Type Classifier

## Sprint Goal

Design and implement per-language dict-based feature extractors (no NLTK deps) that boost sentence-type classification accuracy across 8 languages and fit inside ONNX models.

## In Scope

- Create language-specific feature extraction modules (`features_en.py`, `features_es.py`, ..., `features_nl.py`)
- Each module exports a `LanguageFeatureExtractor` class with per-language rules and keyword dicts
- Implement sklearn-compatible transformer wrapper for pipeline composition (FeatureUnion with TF-IDF)
- Extract categorical features: punctuation signals, first-word intents, lexical metrics, sentence structure hints
- Test on baseline (EN) and measure MI to prune low-signal features
- Replicate validated features to other 7 languages (tuned per language syntax)
- Measure F1 improvement per language (FeatureUnion vs. TF-IDF-only baseline)
- Verify ONNX exportability of final pipeline

## Out of Scope

- POS tagging or external NLP libraries (NLTK, spaCy, etc.)
- Deep learning / embeddings (word2vec, BERT, etc.)
- Retraining on new data or changing the dataset
- Multi-label or hierarchical classification
- GUI or API deployment (focus on model artifact only)
- Cross-lingual transfer learning (each language gets independent rules)

## Success Criteria

- [ ] Per-language feature extractors implemented for all 8 languages
- [ ] SVM + (TF-IDF + categorical features) achieves ≥96% F1 macro on EN test set
- [ ] All other languages show ≥1% F1 improvement vs. TF-IDF-only baseline
- [ ] Feature count bounded to <100 total (TF-IDF + categorical combined)
- [ ] Final pipeline exports to ONNX with no errors
- [ ] Code is documented and reproducible (commit feature sets and extraction rules)
