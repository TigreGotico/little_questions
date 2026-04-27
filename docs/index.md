# little_questions

Classify sentences by **type** (question, command, statement, exclamation, request)
and, for questions, by **expected answer category** (EAT taxonomy: 7 main, 53 fine-grained).

Both classifiers download their ONNX models automatically from HuggingFace on first use.

## Quick start

```python
from little_questions import Sentence

# A question
s = Sentence("Who invented the telephone?")
print(type(s).__name__)          # Question
print(s.sentence_type)           # question
print(s.classification)          # HUM:ind
print(s.main_label)              # HUM
print(s.secondary_label)         # ind
print(s.pretty_label)            # individual (Human)
print(s.confidence)              # 0.94

# A command
s = Sentence("Play some jazz music.")
print(type(s).__name__)          # Command
print(s.sentence_type)           # command

# A statement
s = Sentence("The sky is blue.")
print(type(s).__name__)          # Statement
print(s.sentence_type)           # statement
```

## Sentence types

| Class | `sentence_type` | Example |
|-------|----------------|---------|
| `Question` | `question` | "What is the capital of France?" |
| `Statement` | `statement` | "The sky is blue." |
| `Command` | `command` | "Open the door." |
| `Request` | `request` | "Could you pass the salt?" |
| `Exclamation` | `exclamation` | "What a beautiful day!" |

`Request` subclasses `Command`, so `isinstance(s, Command)` is `True` for requests too.

## EAT taxonomy (question answer types)

| Main | Fine-grained subtypes |
|------|-----------------------|
| `ABBR` | `abb`, `exp` |
| `BOOL` | `yesno` |
| `DESC` | `def`, `desc`, `manner`, `reason` |
| `ENTY` | `animal`, `body`, `color`, `food`, `product`, `sport`, `substance`, … (23 total) |
| `HUM` | `ind`, `gr`, `title`, `desc` |
| `LOC` | `city`, `country`, `state`, `mount`, `water`, … |
| `NUM` | `date`, `money`, `dist`, `count`, `temp`, `speed`, … (13 total) |

## Project layout

```
little_questions/
├── __init__.py      # Sentence, subclasses, get_classifier(), get_scorer(), get_yesno_classifier()
├── classifiers.py   # EatClassifier, SentenceTypeClassifier, YesNoClassifier, _OnnxModel
├── constants.py     # EAT_LABELS_7, EAT_LABELS_53, SENTENCE_TYPES, MAIN_LABEL_NAMES, SEC_LABEL_NAMES
└── models.py        # HF auto-download helpers

train/               # Training-only — install with pip install little-questions[train]
├── classifiers.py       # CalibratedLinearSVCClassifier, LinearSVCClassifier, LogRegClassifier, SGDClassifier, Model2VecClassifier
├── load_eat.py          # EAT dataset loader
├── load_yesno.py        # Yes/no dataset loader
├── train_eat.py         # Train EAT classifiers → ONNX
├── train_eat_m2v.py     # Train Model2Vec EAT variants
├── train_yesno.py       # Train yes/no polarity classifiers → ONNX
├── train_sentence_type.py
├── benchmark_eat.py     # Full EAT benchmark + plots
├── metrics.py           # EvalResult, evaluate(), compare()
├── mlflow_config.py     # MLflow setup
└── push_to_hf.py        # Push models to HuggingFace
```

## Further reading

- [api.md](api.md) — full API reference
- [models.md](models.md) — model files, HuggingFace repos, training
- [classification.md](classification.md) — benchmark results
- [contributing.md](contributing.md) — development setup
