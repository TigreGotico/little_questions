# Status: EAT as Reference — Calibrated Classifiers, M2V, Full HF Replacement

## Checklist

- [x] Add `CalibratedLinearSVCClassifier` to `train/classifiers.py` with `save_onnx()` using `zipmap=False`
- [x] Add `"svm_cal"` to `train/train_eat.py`; update `save_onnx()` helper for calibration options
- [x] Create `train/train_eat_m2v.py` — train 12 M2V variants (2M/8M/32M × plain/+tfidf × 7c/53c)
- [x] Update `train/benchmark_eat.py` — `OnnxScorer.score_batch()`, calibrated `TwoStageOnnxScorer`, `M2VEatScorer`, full benchmark loop, regenerate plots
- [x] Rewrite `little_questions/constants.py` — `EAT_LABELS_7`, `EAT_LABELS_53`, updated label name dicts (add BOOL, remove COSC-only)
- [x] Add `EatClassifier` singleton to `little_questions/classifiers.py`; update/replace `QuestionTypeClassifier`
- [ ] Update `little_questions/__init__.py` — wire `EatClassifier`, add `.classification_scores`, `.confidence`; handle `BOOL:yesno` (no secondary label)
- [ ] Retrain ONNX: `python -m train.train_eat --model svm_cal`
- [ ] Train M2V: `python -m train.train_eat_m2v`
- [ ] Re-benchmark: `python -m train.benchmark_eat` — verify probs sum to 1, plots updated
- [ ] Delete 19 COSC-only scripts from `train/` (see delete list in plan.md)
- [ ] `little-questions-hf`: `git rm -r models/cosc/`; copy ONNX → `models/eat/`; copy m2v → `models/eat_m2v/`; rewrite `manifest.json`; change git remote to `TigreGotico/eat-classifiers`
- [ ] Rewrite `little-questions-hf/README.md` (EAT classifiers; datasets section linking TigreGotico/EAT and TigreGotico/sentence-types-multilingual)
- [ ] Rewrite `little-questions-hf/BENCHMARKS.md` (EAT results only)
- [ ] Create `TigreGotico/eat-classifiers` HF repo and `git push`

## Blockers
<!-- none yet -->
