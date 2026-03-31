# Classification Benchmarks

Benchmarks run with `train/compare_classifiers.py` against a 15% held-out test split (stratified, `random_state=42`). All results logged to MLflow experiment `little-questions-compare`.

## COSC classification (52-class, EN)

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| TF-IDF LinearSVC | ~0.86 | ~0.83 |
| Enhanced SVM (TF-IDF + char + linguistic) | ~0.87 | ~0.84 |
| potion-multilingual-128M + TF-IDF fusion | ~0.82 | ~0.79 |
| PunctuationScorer (baseline) | ~0.20 | ~0.05 |
| HeuristicScorer (baseline) | ~0.45 | ~0.30 |

> Exact per-run metrics are available in MLflow. The table above shows representative values from v0.8.0 training runs.

## COSC classification (6-class, EN)

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| TF-IDF LinearSVC | ~0.94 | ~0.93 |
| potion-base-8M | ~0.91 | ~0.90 |
| potion-base-8M + TF-IDF fusion | ~0.92 | ~0.91 |

## Sentence-type classification (EN)

`SentenceTypeClassifier` trained on a balanced EN dataset (v0.8.0):

| Class | Notes |
|-------|-------|
| question | High precision — ends `?` is a strong signal |
| command | Benefits from linguistic features (imperative verb POS) |
| statement | Most common class |
| exclamation | Rare; benefits from `!` + `what a` / `how JJ` patterns |
| request | Subclass of command; benefits from modal verb detection |

## Running benchmarks

```bash
# Compare all models for EN, 52-class
python -m train.compare_classifiers --lang en --classes 52 --enhanced

# All languages
python -m train.compare_classifiers --all-langs --classes 6 --save-plots
```

Results are saved to `train/reports/` and uploaded to MLflow.
