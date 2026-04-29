# Classification Benchmarks

All EAT benchmarks use a 15% held-out stratified test split (`random_state=42`).
Sentence-type and yes/no benchmarks are evaluated on the full labelled datasets.

## EAT question-type classification (EN)

### Two-stage default (eat7_svm_cal → eat53_svm_cal)

| Metric | Score |
|--------|-------|
| Accuracy | **93.4%** |
| Macro F1 | **93.4%** |

### All baselines

| Model | Classes | Accuracy | Macro F1 |
|-------|---------|----------|----------|
| `eat53_svm_cal` (calibrated SVM) | 53 | 91.3% | 91.2% |
| `eat7_svm_cal` (calibrated SVM) | 7 | 96.3% | 96.0% |
| **Two-stage default** | 53 | **93.4%** | **93.4%** |
| `eat53_svm` (uncalibrated) | 53 | 90.0% | 89.7% |
| `eat53_logreg` | 53 | 88.8% | 88.3% |
| `eat53_sgd` | 53 | 85.4% | 85.0% |
| `m2v-potion-base-32M+tfidf` | 53 | ~91% | ~91% |

Plots saved to `train/reports/eat/`.

## Sentence-type classification

6 classes: `command`, `exclamation`, `polar_question`, `request`, `statement`, `wh_question`.

| Language | Accuracy | Macro F1 |
|----------|----------|----------|
| EN | 99.2% | 99.2% |
| FR | 97.1% | 97.1% |
| IT | 97.0% | 97.0% |
| NL | 98.8% | 98.8% |
| PT | 95.4% | 95.4% |
| DE | 85.6% | 84.9% |
| ES | 74.6% | 72.7% |

Plots saved to `train/reports/sentence_type/`.

## Yes/no answer-polarity classification

3 classes: `yes`, `no`, `maybe`.

| Model | Coverage | Accuracy | Macro F1 |
|-------|----------|----------|----------|
| `yesno_svm_cal_{LANG}` (per-language) | 43 languages | ~91–96% | ~91–96% |
| `yesno_svm_cal_multilingual` (bundled) | all languages | 84.6% | 84.0% |

The bundled multilingual model is used when no language-specific model is available.
Per-language models achieve 90–96% macro F1 on their own language.

Plots saved to `train/reports/yesno/`.

## Running all benchmarks

```bash
pip install little-questions[train]

python -m train.benchmark_eat
python -m train.benchmark_sentence_type
python -m train.benchmark_yesno
```
