# FAQ — little_questions

## Installation

**Q: How do I install it?**
```bash
uv pip install -e /path/to/little_questions
```
Runtime dependencies are declared in `pyproject.toml`: `numpy`, `nltk`, `scikit-learn`, `joblib`, `pyxdg`, `wordfreq`, `requests`.

**Q: Does it need internet access at runtime?**
No — inference is fully offline. Network access happens only once per language, on first use, to download the pre-trained `.onnx` model file (~1–5 MB each) from the GitHub releases page. After that the model is cached in `~/.local/share/little_questions/`.

**Q: Where are the model files cached?**
`~/.local/share/little_questions/` (XDG Base Directory). Override by setting `XDG_DATA_HOME`. See `little_questions/models/__init__.py:LANG2MODEL`.

---

## Basic Usage

**Q: How do I classify a sentence?**
```python
from little_questions import Sentence

s = Sentence("Who invented the telephone?")
print(s.is_question)      # True
print(s.main_label)       # HUM
print(s.secondary_label)  # ind
print(s.pretty_label)     # individual (Human)
print(s.sentence_type)    # question
```

**Q: How do I use a non-English language?**
```python
s = Sentence("¿Quién inventó el teléfono?", model="es")
```
Supported language codes: `en`, `es`, `pt`, `ca`, `fr`, `de`, `it`, `nl`.

**Q: What is `Sentence.parse()`?**
A classmethod added in Phase 2 that provides a discoverable, typed entry point:
```python
s = Sentence.parse("Who are you?", lang="en")
```
Equivalent to `Sentence("Who are you?", model="en")`. Prefer `parse()` in new code.

**Q: What does `Sentence.score` contain?**
A dict of per-type confidence scores from `SentenceScorer.score()`:
```python
{"question": 0.82, "command": 0.0, "statement": 0.0001, "exclamation": 0.0, "request": 0.5}
```

---

## COSC Taxonomy

**Q: What is the COSC taxonomy?**
The COSC (Coarse and Open-ended SEMNLP Challenge) taxonomy from SemEval 2004 / UIUC QC dataset. It classifies questions into 6 main categories (`HUM`, `ENTY`, `DESC`, `NUM`, `LOC`, `ABBR`) and 52 fine-grained subtypes.

**Q: What does `main_label` return?**
One of: `HUM` (Human), `ENTY` (Entity), `DESC` (Description), `NUM` (Numeric), `LOC` (Location), `ABBR` (Abbreviation). Source: `little_questions/__init__.py:Sentence.main_label`.

**Q: What are all 52 subtypes?**

| Main | Subtypes |
|------|----------|
| HUM  | `ind`, `gr`, `title`, `desc` |
| ENTY | `animal`, `body`, `color`, `cremat`, `currency`, `dismed`, `event`, `food`, `instru`, `lang`, `letter`, `other`, `plant`, `product`, `religion`, `sport`, `substance`, `symbol`, `techmeth`, `termeq`, `veh`, `word` |
| DESC | `def`, `desc`, `manner`, `reason` |
| NUM  | `code`, `count`, `date`, `dist`, `money`, `ord`, `other`, `period`, `perc`, `speed`, `temp`, `volsize`, `weight` |
| LOC  | `city`, `country`, `mount`, `other`, `state` |
| ABBR | `abb`, `exp` |

**Q: My sentence is not a question but `main_label` returns something. Why?**
The COSC classifier always predicts a label regardless of sentence type — it was trained on question data. The label is meaningful only when `is_question` is `True`. Check `sentence_type` first.

---

## Sentence Types

**Q: What sentence types are detected?**

| Type | Class | `is_*` property | Trigger |
|------|-------|----------------|---------|
| question | `Question` | `is_question` | wh-word, modal, ends `?` |
| command | `Command` | `is_command` | imperative verb start, no subject |
| request | `Request` | `is_request` | polite modal (`could/would/can`) |
| statement | `Statement` | `is_statement` | S-V-O structure, ends `.` |
| exclamation | `Exclamation` | `is_exclamation` | starts `what`/`how`, ends `!` |

**Q: Is `Request` also a `Command`?**
Yes. `Request` subclasses `Command`, so `is_command` is `True` for requests. Check `is_request` first if you need to distinguish polite requests from direct commands.

**Q: How does sentence type detection work for English?**
A trained `SentenceTypeClassifier` (TF-IDF + M2V embeddings, 93% accuracy) is used for English. The legacy `SentenceScorerHeuristic` (POS-tag based) is kept in `little_questions/classifiers/legacy.py` as a fallback. Source: `little_questions/sentence_type.py:SentenceTypeClassifier`.

**Q: How does sentence type detection work for non-English?**
The fallback `SentenceScorer` uses terminal punctuation heuristics (`?` → question, `!` → exclamation, `.` → command/statement). Source: `little_questions/classifiers/__init__.py:SentenceScorer`.

---

## Classifiers

**Q: Which ML algorithm is used?**
Linear SVM (`sklearn.svm.LinearSVC`) trained on the UIUC QC dataset (4677 balanced questions). Pipeline: TF-IDF (word unigrams/bigrams + char trigrams/4-grams) → `LinearSVC`. Exported to ONNX format for cross-platform inference. Source: `little_questions/classifiers/__init__.py:Classifier`.

**Q: Can I use a different algorithm?**
Training pipelines live in the `train/` package: `LinearSVCClassifier`, `LogRegTextClassifier`, `NaiveBayesTextClassifier`, and others in `train/classifiers.py`. Load a custom model file with `Classifier.load_from_file(path)`.

**Q: How do I retrain a model?**
See `train/`. Use `train/train_en.py` for English or `train/train_all.py` for all languages. Training requires `pip install little_questions[train]` (adds `skl2onnx`, `onnx`). Output is an `.onnx` model file. See `docs/contributing.md` for the full workflow.

**Q: Why is the model cached globally?**
`get_classifier()` in `little_questions/classifiers/__init__.py` keeps a module-level `_LAZY_LOADING` dict so each language model is loaded from disk only once per process. Call `clear_classifier_cache()` to reset (useful in tests or long-running processes that need to free memory).

---

## Errors & Troubleshooting

**Q: `FileNotFoundError` or `OSError` when creating a `Sentence`.**
The model file was not downloaded. Run the download function explicitly:
```python
from little_questions.models import download_en
download_en()
```
Or delete `~/.local/share/little_questions/` and let it re-download on the next `Sentence()` call.

**Q: `ValueError: unknown model` from `get_model_path()`.**
The `model` argument to `Sentence()` is not one of the 7 supported language codes. Check `little_questions.models.LANG2MODEL` for the full list.

**Q: NLTK resource errors (`Resource punkt not found`).**
Run:
```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
```
These are downloaded automatically by `download_en()` but may be missing in fresh environments.

**Q: Tests fail with `ModuleNotFoundError: No module named 'xdg'`.**
Install `pyxdg`: `uv pip install pyxdg`. In CI the `test/conftest.py` stub covers this — the stub patches `xdg.BaseDirectory` so tests run without the package installed.

---

## Development

**Q: How do I run the tests?**
```bash
uv run pytest test/ -v --cov=little_questions --cov-report=term-missing
```

**Q: How do I add a new language?**
1. Create `little_questions/classifiers/lang/<code>/` with `__init__.py` and optionally `postag.py`.
2. Add model filenames to `LANG2MODEL` and download URLs to `MODEL2URL` in `little_questions/models/__init__.py`.
3. Add a `download_<code>()` function in `little_questions/models/__init__.py`.
4. Add the language code to `SUPPORTED_LANGUAGES` in `little_questions/classifiers/__init__.py`.
5. Train the model using `train/train_all.py` as a template.
6. Write smoke tests.

**Q: How do I add a new sentence type?**
Add a subclass of `Sentence`, add the type key to `_TYPE_MAP` in `little_questions/__init__.py`, implement scoring in `SentenceScorerEN` and `SentenceScorer`. Add tests.
