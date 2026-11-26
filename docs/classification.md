# Classification Benchmarks

All EAT benchmarks use a 15% held-out stratified test split (`random_state=42`). The sentence-type and yes/no benchmarks score the models on their full labelled datasets, which include the training data, so these scores are higher than the scores on unseen text.

## EAT question-type classification (EN)

### Two-stage default (eat7_svm_cal → eat53_svm_cal)

| Metric | Score |
|--------|-------|
| Accuracy | **93.4%** |
| Macro F1 | **93.4%** |

### All baselines

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="img/eat-models-dark.svg">
  <img alt="EAT 53-class models on the held-out split: the two-stage default is the highest at 93.4% accuracy; eat53_sgd is the lowest at 84.6%." src="img/eat-models.svg">
</picture>

The figure and its alt text show the 53-class models only. `img/plot_eat_models.py` makes this figure from `train/reports/eat/*_benchmark.json`.

| Model | Classes | Accuracy | Macro F1 |
|-------|---------|----------|----------|
| **Two-stage default** | 53 | **93.4%** | **93.4%** |
| `m2v-potion-base-32M+tfidf` | 53 | 92.2% | 92.0% |
| `eat53_svm` (uncalibrated) | 53 | 91.7% | 91.5% |
| `eat53_svm_cal` (calibrated SVM) | 53 | 91.5% | 91.4% |
| `eat53_logreg` | 53 | 87.7% | 87.8% |
| `eat53_sgd` | 53 | 84.6% | 84.1% |
| `eat7_svm_cal` (calibrated SVM) | 7 | 95.6% | 92.9% |

The values come from the committed `train/reports/eat/*_benchmark.json` files.

Plots saved to `train/reports/eat/`.

## Sentence-type classification

6 classes: `command`, `exclamation`, `polar_question`, `request`, `statement`, `wh_question`. Each language has 9,900 labelled sentences.

| Language | Accuracy | Macro F1 |
| --- | --- | --- |
| EN | 99.2% | 99.2% |
| NL | 98.8% | 98.8% |
| FR | 97.1% | 97.1% |
| IT | 97.0% | 97.0% |
| PT | 95.4% | 95.4% |
| DE | 85.6% | 84.9% |
| ES | 74.6% | 72.7% |

The values come from `train/reports/sentence_type/*_benchmark.json`.

## Yes/no answer-polarity classification

3 classes: `yes`, `no`, `maybe`.

| Model | Languages with a report | Sentences | Accuracy | Macro F1 |
| --- | --- | --- | --- | --- |
| `yesno_svm_cal_{LANG}` (one for each language) | 15 | 200 for each language | 90.0% to 95.5% | 90.1% to 95.5% |
| `yesno_svm_cal_multilingual` (in the package) | all | 8,599 | 84.6% | 84.0% |

The values come from `train/reports/yesno/*_benchmark.json`. The runtime uses the multilingual model when no model exists for the language.

## Running all benchmarks

```bash
uv pip install little-questions[train]
python -m train.benchmark_eat
python -m train.benchmark_sentence_type
python -m train.benchmark_yesno
```

---
[← Models](models.md) · [Home](index.md) · [Contributing →](contributing.md)
