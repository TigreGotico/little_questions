# little_questions

Lightweight multilingual question classifier with COSC taxonomy labelling.
No LLM or network call required at inference time — uses pre-trained sklearn SVM models.

## Supported Languages

| Code | Language   | Model file pattern                    |
|------|------------|---------------------------------------|
| en   | English    | `questions52_svm_EN_0.7.0a1.pkl`     |
| es   | Spanish    | `questions52_svm_ES_googtx_0.7.0a1.pkl` |
| pt   | Portuguese | `questions52_svm_PT_googtx_0.7.0a1.pkl` |
| ca   | Catalan    | `questions52_svm_CA_googtx_0.7.0a1`  |
| fr   | French     | `questions52_svm_FR_googtx_0.7.0a1`  |
| de   | German     | `questions52_svm_DE_googtx_0.7.0a1`  |
| it   | Italian    | `questions52_svm_IT_googtx_0.7.0a1`  |

Models are downloaded on first use from the GitHub releases page and cached in `~/.local/share/little_questions/`.

## COSC Taxonomy

The COSC taxonomy (`Sentence.classification`) uses 6 main categories and 52 fine-grained subtypes.

| Main label | Human-readable | Representative subtypes |
|------------|----------------|------------------------|
| `HUM`      | Human          | `ind` (individual), `gr` (group/org), `title`, `desc` |
| `ENTY`     | Entity         | `body` (organ), `cremat`, `dismed`, `lang`, `veh`, `termeq` |
| `DESC`     | Description    | `def` (definition), `desc`, `manner`, `reason` |
| `NUM`      | Numeric        | `count`, `date`, `dist`, `money`, `ord`, `period`, `perc`, `speed`, `temp`, `volsize`, `weight` |
| `LOC`      | Location       | `city`, `country`, `mount`, `other`, `state` |
| `ABBR`     | Abbreviation   | `abb` (abbreviation), `exp` (expansion) |

`Sentence.main_label` — `little_questions/__init__.py:62`
`Sentence.secondary_label` — `little_questions/__init__.py:68`
`Sentence.pretty_label` — `little_questions/__init__.py:73`

## Sentence Types

Each `Sentence` instance is one of five concrete subclasses:

| Subclass      | `sentence_type` | `is_*` property    |
|---------------|-----------------|--------------------|
| `Question`    | `question`      | `is_question`      |
| `Command`     | `command`       | `is_command`       |
| `Request`     | `request`       | `is_request`       |
| `Exclamation` | `exclamation`   | `is_exclamation`   |
| `Statement`   | `statement`     | `is_statement`     |

`Request` subclasses `Command`, so `is_command` is also `True` for requests.

Classification is performed by `SentenceScorerEN` (English) or the fallback `SentenceScorer`.
Source: `little_questions/classifiers/lang/en/__init__.py`, `little_questions/classifiers/base.py`.

## Usage

```python
from little_questions import Sentence

s = Sentence("What is the capital of France?")
print(s.is_question)       # True
print(s.main_label)        # HUM / ENTY / DESC / NUM / LOC / ABBR
print(s.secondary_label)   # e.g. "ind", "def", "city"
print(s.pretty_label)      # e.g. "definition (Description)"
print(s.sentence_type)     # "question"
print(s.score)             # dict of per-type confidence scores
```

Non-English:

```python
s = Sentence("¿Quién inventó el teléfono?", model="es")
```

## Architecture

```
Sentence.__new__()
├── get_classifier(model_id)   → Classifier (sklearn SVM, joblib)
│     └── Classifier.load_from_file() ← little_questions/classifiers/base.py:88
├── get_scorer(lang)            → SentenceScorer / SentenceScorerEN
│     └── SentenceScorerEN.predict() ← little_questions/classifiers/lang/en/__init__.py:25
└── returns Question / Command / Statement / Exclamation / Request
```

See also:
- `docs/classification.md` — COSC label details
- `docs/models.md` — model download and caching
- `docs/intents.md` — intent integration notes
