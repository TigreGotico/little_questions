# Models
<!-- agent-hint: model file names and sizes come from little_questions/models/; scores are on classification.md. -->

All models are LinearSVC classifiers exported to ONNX with skl2onnx. Inference needs only `onnxruntime`, not scikit-learn.

## Where the models load from

The library looks for each model in this order:

| Location | Contents |
| --- | --- |
| `little_questions/models/` (in the package) | The English EAT and sentence-type models and the multilingual yes/no model |
| `~/.local/share/little_questions/` | Downloaded models and your own models |
| Hugging Face Hub | Each model that is not in the package. The download needs `uv pip install little-questions[hf]`. |

## EAT classifiers (`TigreGotico/eat-classifiers`)

| Model file | Size | Classes | In the package |
| --- | --- | --- | --- |
| `eat7_svm_cal_EN_0.9.0.onnx` | 2.6 MB | 7 | yes |
| `eat53_svm_cal_EN_0.9.0.onnx` | 16 MB | 53 | yes |
| `eat7_svm_cal_unpunct_EN_0.9.0.onnx` | 2.6 MB | 7, for text with no punctuation | yes |
| `eat53_svm_cal_unpunct_EN_0.9.0.onnx` | 16 MB | 53, for text with no punctuation | yes |

[Classification](classification.md) gives the scores of each model.

The default runtime uses the two-stage approach: eat7 predicts the main category, then eat53 scores all 53 labels and renormalises within that category.

### Categorical features

The svm_cal models are trained with lexical signal tokens prepended to the question text before TF-IDF vectorization. The tokens are injected at inference time by `_eat_preprocess()` in `classifiers.py`. No custom ONNX op is needed. The tokens:

| Token | Signal |
| --- | --- |
| `__feat_yesno__` | BOOL (is, does, do, was, are, were, has, will, did) |
| `__feat_who__` | HUM |
| `__feat_where__` | LOC |
| `__feat_when__` | NUM |
| `__feat_why__` | DESC |
| `__feat_howmany__` | NUM (how and a quantifier word) |
| `__feat_define__` | DESC |

## Sentence-type classifiers (`TigreGotico/sentence-types`)

| Model file | Language |
|------------|----------|
| `sentence_type_EN_0.8.0.onnx` | English (in the package) |
| `sentence_type_DE_0.8.0.onnx` | German |
| `sentence_type_ES_0.8.0.onnx` | Spanish |
| `sentence_type_FR_0.8.0.onnx` | French |
| `sentence_type_IT_0.8.0.onnx` | Italian |
| `sentence_type_NL_0.8.0.onnx` | Dutch |
| `sentence_type_PT_0.8.0.onnx` | Portuguese |

## Yes/no classifiers (`TigreGotico/yes-no-classifiers`)

| Model file | Coverage |
|------------|----------|
| `yesno_svm_cal_{LANG}_0.9.0.onnx` | Language-specific (43 languages) |
| `yesno_svm_cal_multilingual_0.9.0.onnx` | All 43 languages combined (in the package) |

The runtime tries the language-specific model first, then falls back to the multilingual model. Character n-gram TF-IDF (2-4) is used. This is language-agnostic and effective for the shallow lexical patterns in yes/no responses.

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
- EAT: `TigreGotico/EAT` (English questions, 53 labels)
- Sentence types: `TigreGotico/sentence-types-multilingual`
- Yes/no: `TigreGotico/yes-no-multilingual` (43 languages)

---
[← API](api.md) · [Home](index.md) · [Classification →](classification.md)
