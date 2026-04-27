# Implementation Notes: Calibrated Confidence Scores for EAT Classifiers

## Patterns to Use

- **`CalibratedClassifierCV` as final pipeline step** — Pipeline([("tfidf", tfidf), ("clf", CalibratedClassifierCV(LinearSVC(...), cv=5, method="sigmoid"))]). The TF-IDF goes first as a named step; the calibrated wrapper replaces the bare LinearSVC.
- **`OnnxClassifier.score()` in classifiers.py:134** — already does softmax on output[1]; for calibrated models output[1] is already a probability matrix, so we skip softmax and read directly.
- **Singleton pattern** — `EatClassifier.get_instance(lang)` with `_instances: dict[str, EatClassifier]` and a threading Lock, exactly like `QuestionTypeClassifier`.

## Gotchas

- **`zipmap=False` is mandatory** — without it, skl2onnx wraps the probability output in a ZipMap ONNX op that returns a sequence of dicts, not an ndarray. Pass `options={CalibratedClassifierCV: {"zipmap": False}}` to `convert_sklearn()`.
- **Output index shifts for calibrated model** — uncalibrated LinearSVC (with `nocl=True`) outputs `[label_int64, decision_scores]`. Calibrated model outputs `[label_int64, probabilities_float32]`. Both are output[0] and output[1], so indexing is the same — but the values are probabilities now, no softmax needed.
- **`cv=5` trains 5 sub-models** — training `svm_cal` takes ~5× longer than plain `svm`. Expected runtime ~2 min for 53-class EN on this hardware.
- **Renormalization after masking** — after zeroing invalid labels in TwoStageOnnxScorer, divide by `surviving_probs.sum()` (not softmax). If the gate model is wrong and all probabilities are zero (edge case), fall back to argmax over original probabilities.
- **Model path convention** — calibrated models use suffix `_cal`: `eat53_svm_cal_EN_0.9.0.onnx`, `eat7_svm_cal_EN_0.9.0.onnx`. The `EatClassifier` in `little_questions/classifiers.py` looks up these paths.
- **`classes` metadata** — still embed the sorted label list as JSON in ONNX metadata under key `"classes"`, same as uncalibrated models.
- **`Sentence.confidence`** — is the max value from `classification_scores`, not a separate ONNX call. Compute once in `__init__` alongside `.classification`.

## Key Imports / APIs

- `from sklearn.calibration import CalibratedClassifierCV` — `train/classifiers.py`
- `from skl2onnx import convert_sklearn` + `options={CalibratedClassifierCV: {"zipmap": False}}` — `train/train_eat.py`
- `ort.InferenceSession` — `little_questions/classifiers.py` (already imported as `ort`)
- `QuestionTypeClassifier` at `little_questions/classifiers.py:~160` — copy singleton pattern exactly

## Conventions

- Model files live in `~/.local/share/little_questions/eat/`
- Version suffix: `0.9.0` (same as uncalibrated baselines)
- `_cal` in model name signals Platt-calibrated output
- `score()` always returns a `dict[str, float]` where values sum to ~1.0
- Keep `svm` (uncalibrated) models intact; `svm_cal` is additive
