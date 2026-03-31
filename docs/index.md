# little_questions

Lightweight multilingual question classifier and sentence-type detector. No LLM required — uses pre-trained sklearn SVM models exported to ONNX.

## Supported languages

| Code | Language   | COSC model | Sentence-type model |
|------|------------|------------|---------------------|
| `en` | English    | ✓ | ✓ (trained) |
| `es` | Spanish    | ✓ | ✓ (trained) |
| `pt` | Portuguese | ✓ | ✓ (trained) |
| `ca` | Catalan    | ✓ | ✓ (trained) |
| `fr` | French     | ✓ | ✓ (trained) |
| `de` | German     | ✓ | ✓ (trained) |
| `it` | Italian    | ✓ | ✓ (trained) |
| `nl` | Dutch      | ✓ | ✓ (trained) |

Models are downloaded on first use from GitHub releases and cached in `~/.local/share/little_questions/`.

## Quick start

```python
from little_questions import Sentence

s = Sentence("What is the capital of France?")
print(s.sentence_type)    # question
print(s.main_label)       # LOC
print(s.secondary_label)  # city
print(s.pretty_label)     # Location — city
print(s.is_question)      # True

# Fast path — COSC label only, no sentence-type overhead
from little_questions import classify, classify_batch
print(classify("Who invented the telephone?"))
print(classify_batch(["What is X?", "Open the door."]))
```

## Architecture

```
little_questions/
├── __init__.py          # Sentence, classify(), classify_batch(), subclasses
├── classifiers/
│   └── __init__.py      # Classifier (ONNX/joblib), get_classifier(), get_scorer()
├── sentence_type.py     # SentenceTypeClassifier (TF-IDF + LinearSVC)
├── models/
│   └── __init__.py      # download(), get_model_path(), LANG2MODEL, MODEL2URL
├── constants.py         # SUPPORTED_LANGUAGES, SENTENCE_TYPES
└── version.py           # version block

train/                   # Training-only (not installed with the package)
├── classifiers.py       # LinearSVCClassifier, Model2VecClassifier, …
├── baselines.py         # PunctuationScorer, HeuristicScorer (benchmarking only)
├── features.py          # LinguisticFeaturesTransformer (40+ POS/lexical features)
├── metrics.py           # EvalResult, evaluate(), compare()
├── mlflow_config.py     # MLflow tracking helpers
├── tune.py              # Optuna HPO + SGD per-epoch training
├── train_all.py         # Train all language SVM models
└── compare_classifiers.py  # Benchmark SVM vs Potion vs baselines
```

## COSC taxonomy

52 fine-grained question types grouped into 6 coarse categories.

| Coarse | Subtypes |
|--------|----------|
| `ABBR` | `abb` `exp` |
| `DESC` | `def` `desc` `manner` `reason` |
| `ENTY` | `animal` `body` `color` `cremat` `currency` `dismed` `event` `food` `instru` `lang` `letter` `other` `plant` `product` `religion` `sport` `substance` `symbol` `techmeth` `termeq` `veh` `word` |
| `HUM`  | `desc` `gr` `ind` `title` |
| `LOC`  | `city` `country` `mount` `other` `state` |
| `NUM`  | `code` `count` `date` `dist` `money` `ord` `other` `perc` `period` `speed` `temp` `volsize` `weight` |

## Sentence types

| Subclass | `sentence_type` | Notes |
|----------|-----------------|-------|
| `Question` | `question` | |
| `Command` | `command` | |
| `Request` | `request` | Subclass of `Command`; also satisfies `is_command` |
| `Statement` | `statement` | |
| `Exclamation` | `exclamation` | |

## Further reading

- [api.md](api.md) — full API reference
- [models.md](models.md) — model files and download helpers
- [classification.md](classification.md) — benchmark results
- [contributing.md](contributing.md) — development setup and training workflow
