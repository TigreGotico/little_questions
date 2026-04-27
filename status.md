# Status: EAT as Reference — Calibrated Classifiers, M2V, Full HF Replacement

## Checklist

- [x] Add `CalibratedLinearSVCClassifier` to `train/classifiers.py` with `save_onnx()` using `zipmap=False`
- [x] Add `"svm_cal"` to `train/train_eat.py`; update `save_onnx()` helper for calibration options
- [x] Create `train/train_eat_m2v.py` — train 12 M2V variants (2M/8M/32M × plain/+tfidf × 7c/53c)
- [x] Update `train/benchmark_eat.py` — `OnnxScorer.score_batch()`, calibrated `TwoStageOnnxScorer`, `M2VEatScorer`, full benchmark loop, regenerate plots
- [x] Rewrite `little_questions/constants.py` — `EAT_LABELS_7`, `EAT_LABELS_53`, updated label name dicts (add BOOL, remove COSC-only)
- [x] Add `EatClassifier` singleton to `little_questions/classifiers.py`; update/replace `QuestionTypeClassifier`
- [x] Update `little_questions/__init__.py` — wire `EatClassifier`, add `.classification_scores`, `.confidence`; handle `BOOL:yesno` (no secondary label)
- [ ] Retrain ONNX: `python -m train.train_eat --model svm_cal`
- [ ] Train M2V: `python -m train.train_eat_m2v`
- [ ] Re-benchmark: `python -m train.benchmark_eat` — verify probs sum to 1, plots updated
- [x] Delete COSC-only scripts from `train/` — removed 15; kept translate.py/translate_dataset.py/translate_cosc.py/translate_cosc_batch.py for future multilingual EAT use
- [x] Delete `little-questions-hf/` monorepo — replaced by one-HF-repo-per-model approach
- [x] Create `train/push_to_hf.py` — pushes each ONNX/M2V model to its own HF repo with generated README model card
- [ ] Run `python -m train.push_to_hf --type onnx` after ONNX training
- [ ] Run `python -m train.push_to_hf --type m2v` after M2V training

## Blockers
<!-- none yet -->
