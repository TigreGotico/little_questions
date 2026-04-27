# Classification Benchmarks

All benchmarks use a 15% held-out stratified test split (`random_state=42`) and are logged to MLflow experiment `EAT`.

## EAT question-type classification (EN)

### Two-stage default (eat7_svm_cal → eat53_svm_cal)

| Metric | Score |
|--------|-------|
| Accuracy | — |
| Macro F1 | **93.4%** |

### All baselines

| Model | Classes | Accuracy | Macro F1 |
|-------|---------|----------|----------|
| `eat53_svm_cal` (calibrated SVM) | 53 | — | 91.0% |
| `eat7_svm_cal` (calibrated SVM) | 7 | — | 95.6% |
| Two-stage default | 53 | — | **93.4%** |
| `eat53_svm` (uncalibrated) | 53 | — | — |
| `eat53_logreg` | 53 | — | — |
| `eat53_sgd` | 53 | — | — |
| `m2v-potion-base-8M+tfidf` | 53 | — | ~91–92% |

Run `python -m train.benchmark_eat` to generate plots and populate the table with exact numbers.

Plots are saved to `train/reports/eat/`:
- `benchmark_eat53_overview.png` — all baselines, accuracy + macro F1
- `benchmark_eat53_confusion.png` — normalised confusion matrix (best model)
- `benchmark_eat53_per_class_f1.png` — per-class F1, sorted ascending
- `benchmark_eat7_confusion.png` — 7-class confusion matrix
- `benchmark_eat_model_comparison.png` — macro F1 line plot, 53-class vs 7-class

## Sentence-type classification

Run `python -m train.benchmark_sentence_type` to generate numbers and plots.

Supported languages with ONNX models: `en`, `de`, `es`, `fr`, `it`, `nl`, `pt`.

Plots saved to `train/reports/sentence_type/`:
- `benchmark_sentence_type_overview.png` — accuracy + macro F1 per language
- `benchmark_sentence_type_{lang}_confusion.png` — per-language confusion matrices

## Yes/no polarity classification

Run `python -m train.train_yesno --plot` to generate numbers and overview chart.

| Model | Coverage | Macro F1 |
|-------|----------|----------|
| `yesno_svm_cal_multilingual` | 43 languages | — |
| `yesno_svm_cal_{LANG}` | per-language | — |

Plot saved to `train/reports/yesno/benchmark_yesno_overview.png`.

## Running all benchmarks

```bash
pip install little-questions[train]

python -m train.benchmark_eat
python -m train.train_yesno --plot
```
