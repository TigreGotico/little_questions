# Maintenance Report — little_questions

## 2026-03-31

**AI Model**: Claude Sonnet 4.6

**Actions Taken**:
- Augmented `train/clean_data/sentence_types_EN.csv` and `sentence_types_EN.txt` from ~3,001 entries to 9,900 entries
- All 6 classes balanced at exactly 1,650 samples each: command, exclamation, polar_question, request, statement, wh_question
- Deduplication enforced throughout via lowercase set
- Updated `FAQ.md` with dataset documentation

**Oversight**: AI-generated entries reviewed for label correctness against the classification rules defined in the plan. Boundary cases (polar_question vs wh_question, request vs command) manually verified by spot-check.

---
