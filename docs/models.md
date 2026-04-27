# Models

All models are LinearSVC classifiers exported to ONNX via skl2onnx.
Inference requires only `onnxruntime` — no sklearn at runtime.
Models are downloaded on first use from HuggingFace and cached in
`~/.local/share/little_questions/`.

## EAT classifiers — `TigreGotico/eat-classifiers`

| Model file | Size | Accuracy | Notes |
|------------|------|----------|-------|
| `eat53_svm_cal_EN_0.9.0.onnx` | 16 MB | 91.0% macro F1 (53-class) | Calibrated probabilities |
| `eat7_svm_cal_EN_0.9.0.onnx` | 2.6 MB | 95.6% macro F1 (7-class) | Calibrated probabilities |
| **Two-stage default** | 18.6 MB total | **93.4%** macro F1 (53-class) | eat7 gates eat53 |
| `eat53_svm_cal_asr_EN_0.9.0.onnx` | 16 MB | — | Unpunctuated/ASR variant |
| `eat7_svm_cal_asr_EN_0.9.0.onnx` | 2.6 MB | — | Unpunctuated/ASR variant |

The default runtime uses the two-stage approach: eat7 predicts the main category,
then eat53 scores all 53 labels and renormalises within that category.

### Categorical features

The svm_cal models are trained with high-precision lexical signal tokens prepended
to the question text before TF-IDF vectorization. The tokens are injected at
inference time by `_eat_preprocess()` in `classifiers.py` — no custom ONNX op
is needed. Token precision on the EAT corpus:

| Token | Signal | Precision |
|-------|--------|-----------|
| `__feat_yesno__` | BOOL | 100% (is/does/do/was/are/were/has/will/did) |
| `__feat_who__` | HUM | 86% |
| `__feat_where__` | LOC | 99% |
| `__feat_when__` | NUM | 94% |
| `__feat_why__` | DESC | 100% |
| `__feat_howmany__` | NUM | ~90% (how + quantifier word) |
| `__feat_define__` | DESC | 100% |

## Sentence-type classifiers — `TigreGotico/sentence-types`

| Model file | Language |
|------------|----------|
| `sentence_type_EN_0.8.0.onnx` | English |
| `sentence_type_DE_0.8.0.onnx` | German |
| `sentence_type_ES_0.8.0.onnx` | Spanish |
| `sentence_type_FR_0.8.0.onnx` | French |
| `sentence_type_IT_0.8.0.onnx` | Italian |
| `sentence_type_NL_0.8.0.onnx` | Dutch |
| `sentence_type_PT_0.8.0.onnx` | Portuguese |

## Yes/no classifiers — `TigreGotico/yes-no-classifiers`

| Model file | Coverage |
|------------|----------|
| `yesno_svm_cal_{LANG}_0.9.0.onnx` | Language-specific (43 languages) |
| `yesno_svm_cal_multilingual_0.9.0.onnx` | All 43 languages combined |

The runtime tries the language-specific model first, then falls back to the
multilingual model. Character n-gram TF-IDF (2–4) is used — language-agnostic
and effective for the shallow lexical patterns in yes/no responses.

## Training

```bash
pip install little-questions[train]

# EAT classifiers (ONNX baselines)
python -m train.train_eat                    # all baselines, 53-class + 7-class
python -m train.train_eat --model svm_cal    # calibrated SVM only

# EAT Model2Vec variants (12 models)
python -m train.train_eat_m2v

# Yes/no polarity classifiers
python -m train.train_yesno                  # all languages + multilingual

# Sentence-type classifiers
python -m train.train_sentence_type

# Full EAT benchmark + plots
python -m train.benchmark_eat

# Push to HuggingFace
python -m train.push_to_hf
```

Training data:
- EAT: `TigreGotico/EAT` (30K EN questions, 53 labels)
- Sentence types: `TigreGotico/sentence-types-multilingual` (80K multilingual)
- Yes/no: `TigreGotico/yes-no-multilingual` (8.6K, 43 languages)
