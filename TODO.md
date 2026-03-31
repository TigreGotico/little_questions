# TODO — little_questions

## In Progress / Next

- [ ] Translate `sentence_types_EN.csv` to ES, FR, DE, IT, PT, CA, NL using a controlled pipeline
- [ ] Retrain sentence-type ONNX models for EN with the expanded 9,900-entry dataset
- [ ] Evaluate accuracy delta vs previous model (target: macro F1 ≥ 0.92)
- [ ] Export retrained models via `train/train_all.py` and collect with `collect_models.py`

## Completed

- [x] Split `question` class into `polar_question` and `wh_question`
- [x] Augment EN dataset to 1,650 samples per class (9,900 total) — 2026-03-31
- [x] Validate label boundaries (request vs question, polar vs wh)
- [x] Phase 1 revival: pyproject.toml, refactor, 28 unit tests, docs — fe68040
- [x] Expand EN dataset to 3,001 sentences with legacy heuristic fallback — 5c8e353
