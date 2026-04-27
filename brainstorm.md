# Brainstorm: Language-Specific Dict Feature Extractors for Sentence-Type Classifier

## Problem Statement

The sentence-type classifier (6 labels: command, exclamation, polar_question, request, statement, wh_question) uses TF-IDF + SVM, achieving 95% on English but dropping to 90% on Spanish and other languages. The existing `train/features.py` relies on NLTK, blocking ONNX embedding. 

**Goal**: Build a **per-language dict-based feature extractor** (no external NLP deps) that outputs categorical features. Each language gets its own feature extraction rules and keyword dicts, plugged into sklearn pipeline via DictVectorizer. Result: lightweight, ONNX-friendly, improvable per language.

## Data

- HuggingFace [TigreGotico/sentence-types-multilingual](https://huggingface.co/datasets/TigreGotico/sentence-types-multilingual)
- 8 languages: EN, ES, CA, FR, DE, IT, NL, PT (some machine-translated)
- Labels: command, exclamation, polar_question, request, statement, wh_question
- Baseline: LinearSVC + TF-IDF only (EN: 95%, ES: 90%)

## Ideas & Approaches

- **Per-Language Feature Dicts** — Create `features_en.py`, `features_es.py`, ... with language-specific rules and keyword lists
- **DictVectorizer Transformer** — Wrap dict extraction in sklearn transformer for pipeline composition with TF-IDF via FeatureUnion
- **Multilingual Keywords** — Each lang gets WH_STARTERS, POLITE_WORDS, COMMAND_VERBS, EXCLAMATION_MARKERS, etc. tailored to syntax
- **Binary Intent Flags** — Extract: starts_wh, starts_polite, ends_question, ends_exclamation, starts_command, has_negation
- **Lexical Features** — Sentence length, unique word count, lexical diversity, avg token length
- **Punctuation Signals** — Count ?, !, ..., multiple punctuation, etc.
- **Morphological Hints** — For non-English: suffix patterns, particle markers, or capitalization heuristics
- **Ablation Testing** — Test TF-IDF alone vs. (TF-IDF + categorical features) vs. categorical features alone per language

## Constraints

- **No NLTK / POS tagging** — Simple regex or `.split()` tokenization only
- **No external files at inference** — Constants hardcoded or in `.py` files, not JSON (must be in ONNX)
- **ONNX-exportable** — Output must be sklearn-friendly feature matrix
- **Language-agnostic interface** — `sentence_to_features(text, lang)` works for all 8 languages
- **Self-contained in ONNX** — No runtime lookups or API calls

## Open Questions

1. Should categorical features be *added* to TF-IDF (FeatureUnion) or replace it?
2. How many keywords/rules per language is reasonable? (10 lists? 100 rules?)
3. Which labels are hardest to separate per language? Should we focus feature engineering on those?
4. Should we test EN first, then replicate to other languages, or prototype all 8 in parallel?

## Risks

- **Manual keyword tuning overhead** — Creating 8 language-specific feature sets is labor-intensive
- **Overfitting to EN** — EN features may not translate to machine-translated data
- **Diminishing returns** — TF-IDF likely captures 90%+ of signal; categorical features may add <2% F1
- **Maintenance burden** — Per-language rules become technical debt if not systematized

**Mitigations:**
- Start with EN, measure MI for each feature, prune low-signal ones
- Test on all 8 languages after each iteration
- Use a template approach: define feature extraction once, parameterize by language
- Commit feature sets to repo for reproducibility and iteration
