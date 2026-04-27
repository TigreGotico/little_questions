# Plan: EAT as Reference — Calibrated Classifiers, M2V, Full HF Replacement

## Approach

COSC is fully deprecated. EAT (`TigreGotico/EAT`, 30K samples, 53 labels across 7 main categories including `BOOL`) becomes the single reference taxonomy for question-type classification. All downstream code — constants, classifiers, HF repo, README — switches to EAT labels. No COSC reference survives anywhere.

Training produces two model families on EAT:
1. **ONNX-only** — `CalibratedClassifierCV(LinearSVC, sigmoid, cv=5)` at both 53-class and 7-class granularity. Platt calibration bakes into the ONNX graph; output[1] is a true probability vector. Two-stage scorer (eat7→eat53) masks + renormalizes for a valid conditional distribution.
2. **M2V** — `Model2VecClassifier` with potion-base-2M/8M/32M (EN-only, since EAT is EN-only), with and without TF-IDF fusion, at both 7-class and 53-class — 12 joblib models. These expose `score()` returning calibrated-ish probabilities via `predict_proba` if available, or softmax over decision scores.

The `little_questions` runtime switches to `EatClassifier` (calibrated ONNX) with `score() -> dict[str,float]` and `.confidence` on `Sentence`. `constants.py` is rewritten for EAT. A **new** HF model repo (`TigreGotico/eat-classifiers`) is created for all EAT question-type models; the existing `TigreGotico/sentence-types` repo is left untouched. The local `little-questions-hf/` directory is repurposed as the working tree for the new repo (old cosc models deleted, new eat models added). README and BENCHMARKS are written fresh for the new repo.

## Architecture / Data Flow

```
TigreGotico/EAT (HuggingFace dataset)
  └─ load_eat_hf(classes=53|7)

train/train_eat.py
  CalibratedLinearSVCClassifier → eat53_svm_cal_EN_0.9.0.onnx
                                  eat7_svm_cal_EN_0.9.0.onnx
  (existing svm/logreg/sgd remain for benchmark comparison)

train/train_eat_m2v.py  (new)
  Model2VecClassifier × {2M, 8M, 32M} × {plain, +tfidf} × {53c, 7c}
  → models/eat_m2v/m2v-potion-base-{2M,8M,32M}[-+tfidf]-en-{53,7}c/
      clf.joblib, [tfidf.joblib], model2vec/, meta.json

train/benchmark_eat.py  (updated)
  OnnxScorer.score_batch()          ← proba from output[1]
  TwoStageOnnxScorer.score_batch()  ← mask + renormalize proba
  M2VEatScorer (new)                ← predict + score for m2v models
  All scorers benchmarked, comparison plots regenerated

little_questions/constants.py  (updated)
  EAT_LABELS_7   = ["ABBR","BOOL","DESC","ENTY","HUM","LOC","NUM"]
  EAT_LABELS_53  = [53 fine-grained EAT labels]
  MAIN_LABEL_NAMES  ← add BOOL; remove old COSC-only entries
  SEC_LABEL_NAMES   ← add yesno; prune unused

little_questions/classifiers.py  (updated)
  EatClassifier (singleton, ONNX calibrated, predict + score)
  Remove / replace QuestionTypeClassifier COSC references

little_questions/__init__.py  (updated)
  Sentence.classification        ← EAT label (e.g. "HUM:ind")
  Sentence.main_label            ← e.g. "HUM"
  Sentence.secondary_label       ← e.g. "ind"
  Sentence.classification_scores ← dict[str,float] calibrated proba
  Sentence.confidence            ← float

little-questions-hf/   (repurposed → TigreGotico/eat-classifiers, NEW repo)
  models/cosc/    ← DELETED entirely (git rm -r)
  models/eat/     ← eat53_svm_cal_EN_0.9.0.onnx, eat7_svm_cal_EN_0.9.0.onnx
  models/eat_m2v/ ← 12 m2v model directories
  manifest.json   ← cosc_svm + cosc_m2v removed; eat_svm_cal + eat_m2v added
  README.md       ← rewritten for eat-classifiers repo (EAT models only)
  BENCHMARKS.md   ← EAT results only
  git remote set-url origin https://huggingface.co/TigreGotico/eat-classifiers
  → git push → TigreGotico/eat-classifiers (new repo, created via HF API/web first)

TigreGotico/sentence-types  ← UNTOUCHED (sentence-type models stay there)
```

## Scripts: Delete vs Keep

**Delete** (COSC question-type only — sentence-type scripts are untouched):
- `train/train_all.py` — trains COSC classifiers for all languages
- `train/train_en.py` — trains English COSC classifier
- `train/translate_cosc.py` — translates COSC dataset
- `train/translate_cosc_batch.py` — batch translates COSC dataset
- `train/tune.py` — Optuna hyperparameter search for COSC
- `train/compare_classifiers.py` — benchmarks COSC TF-IDF vs m2v vs heuristic
- `train/benchmark_plots.py` — generates COSC-era benchmark plots from `reports/*.txt`
- `train/validate_exported_models.py` — validates COSC pickle models
- `train/test_m2v.py` — COSC m2v test script
- `train/test_m2v_models.py` — COSC m2v model tests
- `train/utils.py` — `load_data()` for old COSC format (replaced by `load_eat.py`)
- `train/features.py` — legacy NLTK-based feature extraction (COSC-era, not used by sentence-type pipeline)
- `train/test_embeddings.py` — COSC embedding tests
- `train/test_features.py` — COSC feature tests
- `train/test_pos.py` — COSC POS tests
- `train/test_potion.py` — COSC potion model tests
- `train/plot_categorical_impact.py` — COSC categorical feature impact plots
- `train/translate_dataset.py` — COSC-era generic translation helper
- `train/translate.py` — same

**Keep / update**:
- `train/classifiers.py` — add `CalibratedLinearSVCClassifier`; keep `Model2VecClassifier`
- `train/train_eat.py` — add `svm_cal`
- `train/train_eat_m2v.py` — new
- `train/benchmark_eat.py` — updated
- `train/load_eat.py` — keep as-is
- `train/metrics.py` — keep as-is (generic)
- `train/mlflow_config.py` — keep as-is (generic)
- `train/baselines.py` — keep (generic, used in EAT benchmark)
- `train/train_sentence_type.py` — keep (sentence-type, unrelated to COSC)
- `train/train_sentence_type_categorical.py` — keep (sentence-type)
- `train/eval_sentence_type.py` — keep (sentence-type)
- `train/eval_sentence_type_categorical.py` — keep (sentence-type)
- `train/export_sentence_type_onnx.py` — keep (sentence-type export)
- `train/export_onnx_hybrid.py` — keep (sentence-type hybrid ONNX export)
- `train/lang/` — keep (sentence-type categorical feature extractors)

## Implementation Steps

1. **`train/classifiers.py`** — add `CalibratedLinearSVCClassifier`: `Pipeline([("tfidf", TfidfVectorizer(...)), ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000), cv=5, method="sigmoid"))])`. Override `save_onnx()` with `options={CalibratedClassifierCV: {"zipmap": False}}`.

2. **`train/train_eat.py`** — add `"svm_cal"` to `BASELINES` using `CalibratedLinearSVCClassifier`; update `save_onnx()` helper to detect and pass calibration options.

3. **`train/train_eat_m2v.py`** (new) — train all 12 M2V variants on EAT. Reuse `Model2VecClassifier` from `train/classifiers.py`. Matrix: `{potion-base-2M, 8M, 32M}` × `{plain, +tfidf}` × `{53c, 7c}`. Save with `clf.save(directory)`. Log to MLflow `"little-questions-eat"` experiment.

4. **`train/benchmark_eat.py`** — (a) add `score_batch()` to `OnnxScorer`; (b) update `TwoStageOnnxScorer` to use calibrated models, mask + renormalize proba, expose `score_batch()`; (c) add `M2VEatScorer` wrapping `Model2VecClassifier.load()`; (d) add `"svm_cal"`, `"2stage_svm_cal"`, and all m2v variants to benchmark loop; (e) regenerate all plots.

5. **`little_questions/constants.py`** — replace `QUESTION_TYPES_6` / `QUESTION_TYPES` with `EAT_LABELS_7` / `EAT_LABELS_53` using the full EAT taxonomy (7 main incl. `BOOL`, 53 fine-grained incl. `BOOL:yesno`). Update `MAIN_LABEL_NAMES` (add `BOOL: "Boolean"`) and `SEC_LABEL_NAMES` (add `yesno`, `manner`, `reason`; remove COSC-only entries that don't appear in EAT).

6. **`little_questions/classifiers.py`** — add `EatClassifier` singleton (load `eat53_svm_cal_{LANG_UPPER}_0.9.0.onnx`, `predict() -> str`, `score() -> dict[str,float]`). Update or replace `QuestionTypeClassifier` to delegate to `EatClassifier` for backward compat, or rename cleanly.

7. **`little_questions/__init__.py`** — wire `EatClassifier` into `Sentence.__init__`; add `.classification_scores` and `.confidence` properties; ensure `.main_label` and `.secondary_label` work with EAT labels (including `BOOL:yesno` which has no secondary).

8. **Retrain ONNX** — `python -m train.train_eat --model svm_cal` (produces 2 ONNX files).

9. **Train M2V** — `python -m train.train_eat_m2v` (produces 12 model directories).

10. **Re-benchmark** — `python -m train.benchmark_eat` — verify probs sum to 1, 2-stage confidence tracks accuracy, m2v results tabulated; all plots saved to `train/reports/eat/`.

11. **Delete COSC train scripts** — `git rm` the 19 COSC-only scripts listed above; commit.

12. **`little-questions-hf` repurpose** — `git rm -r models/cosc/`; copy ONNX into `models/eat/`; copy m2v dirs into `models/eat_m2v/`; rewrite `manifest.json` (only `eat_svm_cal` + `eat_m2v` keys); copy updated benchmark plots; change git remote: `git remote set-url origin https://huggingface.co/TigreGotico/eat-classifiers`.

13. **Rewrite `little-questions-hf/README.md`** — EAT question-type classifiers (calibrated ONNX + m2v); usage examples with EAT labels; dedicated **Datasets** section linking both source datasets built for this project: [`TigreGotico/EAT`](https://huggingface.co/datasets/TigreGotico/EAT) (question-type taxonomy, 30K samples) and [`TigreGotico/sentence-types-multilingual`](https://huggingface.co/datasets/TigreGotico/sentence-types-multilingual) (sentence-type taxonomy, 80K multilingual samples); no COSC anywhere.

14. **Rewrite `little-questions-hf/BENCHMARKS.md`** — full EAT results table (all baselines + 2-stage + m2v variants); no COSC.

15. **Create HF repo + push** — create `TigreGotico/eat-classifiers` on HuggingFace (via `huggingface_hub.create_repo()` or web UI), then `git push -u origin main` in `little-questions-hf/`.
