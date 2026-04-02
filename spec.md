# Spec: Language-Specific Categorical Feature Extractors for Sentence-Type Classifier

## Objective

Build per-language dict-based feature extractors (no external NLP dependencies) that extract categorical signals from raw text for sentence-type classification across 7 languages (EN, ES, FR, DE, IT, NL, PT). Features must be sklearn-compatible transformers composable with TF-IDF via FeatureUnion, boost F1 scores on test sets, and export cleanly to ONNX models.

**Note:** CA (Catalan) was removed from scope — the CA dataset is too small and noisy for reliable training (high label noise, insufficient examples per class). See decisions.md for the full rationale.

## Functional Requirements

1. **Create language-specific feature modules** — Implement `train/lang/features_<lang>.py` stub wrappers for each of EN, ES, FR, DE, IT, NL, PT; each re-exports `LanguageFeatureExtractor` from the consolidated `train/lang/feature_extractors.py` module. Each class defines keyword dicts and extraction rules tailored to that language's syntax.

2. **Extract categorical intent signals** — For each sentence, extract binary flags: `starts_wh` (wh-question), `starts_polite` (polite modal), `starts_command` (imperative verb), `ends_question`, `ends_exclamation`, `has_negation`.

3. **Extract lexical & punctuation features** — Compute: sentence length, token count, unique token count, lexical diversity, avg token length, question mark count, exclamation mark count, ellipsis count.

4. **Implement sklearn transformer wrapper** — Create `LanguageFeatureTransformer` class (fit/transform interface) that accepts language code, outputs dense or sparse feature matrix via DictVectorizer, works in Pipeline and FeatureUnion.

5. **Compose with TF-IDF in FeatureUnion** — Test two pipelines: (a) TF-IDF only (baseline), (b) FeatureUnion([TF-IDF, categorical features]) with SVM; measure F1 improvement per language.

6. **Measure feature importance** — Compute mutual information (MI) for each categorical feature on EN dataset; prune features with MI < 0.01 bits.

7. **Bound feature count** — Ensure total feature count (TF-IDF vocabulary + categorical) ≤ 100; if exceeded, reduce TF-IDF vocabulary or prune low-MI categorical features.

8. **Export final pipeline to ONNX** — Use sklearn2onnx to convert fitted Pipeline (LinearSVC + FeatureUnion) to ONNX format; verify model runs and produces identical predictions.

9. **Document extraction rules** — For each language, commit a summary of keywords, intent signals, and heuristics; explain why each rule is language-specific.

10. **Test on all 8 languages** — Run evaluation script after each language implementation; report F1 macro, weighted F1, per-class precision/recall on held-out test sets (15% of data).

## Non-Goals

- Use POS tagging, lemmatization, or any external NLP library (NLTK, spaCy, CoreNLP, etc.)
- Implement deep learning models, embeddings (Word2Vec, BERT, fastText), or neural networks
- Retrain on new data, augment the existing dataset, or change label definitions
- Support multi-label classification or hierarchical sentence-type taxonomy
- Build GUI, REST API, or production deployment infrastructure
- Achieve cross-lingual transfer (each language gets independent feature rules)
- Optimize for latency, throughput, or model size (focus on accuracy only)
- Generate feature interpretability reports or SHAP explanations
- Benchmark against alternative ML algorithms (Random Forest, Gradient Boosting, etc.)

## Interfaces & Contracts

- **Input**: Raw text string (any language code EN/ES/FR/DE/IT/NL/PT)
- **Output**: Feature matrix (sklearn-compatible: dense or sparse ndarray), shape `(n_samples, n_features)`
- **Transformer API**: `fit(X, y)` → `self`, `transform(X)` → feature matrix, `fit_transform(X, y)` → feature matrix (sklearn standard)
- **Feature dict keys**: All lowercase, snake_case (e.g., `starts_wh`, `lexical_diversity`, `ends_exclamation`)
- **DictVectorizer**: Sparse matrix output by default; dense on demand via `transformer.sparse=False`
- **ONNX model input**: `input` (string, shape `[batch_size]`)
- **ONNX model output**: Decision function scores (logits) + predicted class index
- **Dataset loader**: `train/utils.py::load_data(path, classes=6)` returns `(texts, labels)` for 6-class sentence types

## Acceptance Criteria

- [ ] Feature modules `features_en.py` through `features_nl.py` created and importable
- [ ] Each module exports `LanguageFeatureExtractor` class with `extract(text: str) -> Dict[str, float]` method
- [ ] `LanguageFeatureTransformer` sklearn wrapper passes `set_params()` and `get_params()` tests
- [ ] EN baseline achieves ≥96% F1 macro on test set with (TF-IDF + categorical features) vs. SVM
- [ ] ES, FR, DE, IT, NL, PT each show ≥1% absolute F1 improvement vs. TF-IDF-only baseline (CA removed — see note above)
- [ ] Feature count (TF-IDF vocab + categorical features) ≤ 100 on EN dataset
- [ ] Mutual information measured and low-signal features (MI < 0.01) pruned from EN extractor
- [ ] TF-IDF+LinearSVC core pipeline exports to ONNX without errors using sklearn2onnx; categorical features computed at inference via sklearn wrapper (hybrid approach — `LanguageFeatureTransformer` wraps `DictVectorizer` which skl2onnx cannot convert, so the ONNX model covers the TF-IDF+SVC path only; full-accuracy models saved as pickle)
- [ ] ONNX model inference produces identical class predictions as TF-IDF-only sklearn Pipeline on 100 random test samples per language (categorical features excluded from ONNX by design)
- [ ] All feature extraction rules and language-specific keywords documented in `train/features_<lang>.py` docstrings
- [ ] Test reports generated: `train/reports/sentence_type_<lang>_categorical_features_0.8.0.txt` for each language, showing F1 improvement
