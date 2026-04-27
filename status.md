# Status: EAT as Reference — Calibrated Classifiers, M2V, Full HF Replacement

## Checklist

- [x] Add `CalibratedLinearSVCClassifier` to `train/classifiers.py` with `save_onnx()` using `zipmap=False`
- [x] Add `"svm_cal"` to `train/train_eat.py`; update `save_onnx()` helper for calibration options
- [x] Create `train/train_eat_m2v.py` — train 12 M2V variants (2M/8M/32M × plain/+tfidf × 7c/53c)
- [x] Update `train/benchmark_eat.py` — `OnnxScorer.score_batch()`, calibrated `TwoStageOnnxScorer`, `M2VEatScorer`, full benchmark loop, regenerate plots
- [x] Rewrite `little_questions/constants.py` — `EAT_LABELS_7`, `EAT_LABELS_53`, updated label name dicts (add BOOL, remove COSC-only)
- [x] Add `EatClassifier` singleton to `little_questions/classifiers.py`; update/replace `QuestionTypeClassifier`
- [x] Update `little_questions/__init__.py` — wire `EatClassifier`, add `.classification_scores`, `.confidence`; handle `BOOL:yesno` (no secondary label)
- [x] Retrain ONNX: all 8 models trained (svm/logreg/sgd/svm_cal × 53c/7c)
- [x] Train M2V: all 12 variants trained (2M/8M/32M × plain/+tfidf × 53c/7c)
- [x] Re-benchmark: all 22 scorers evaluated, 7 plots saved to `train/reports/eat/`
- [x] Delete COSC-only scripts from `train/` — removed 15; kept translate scripts for future multilingual EAT use
- [x] Delete `little-questions-hf/` — replaced by single-repo `TigreGotico/eat-classifiers` approach
- [x] Create `train/push_to_hf.py` — uploads all ONNX + M2V models + benchmark reports to `TigreGotico/eat-classifiers` with auto-generated README + BENCHMARKS.md
- [ ] Run `python -m train.push_to_hf` (requires HF login)

## Blockers
<!-- none yet -->
