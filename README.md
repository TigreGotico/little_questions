# little_questions

Classify sentences by **type** and, for questions, by **expected answer category**.

Three classifiers, all ONNX-backed, all offline out of the box:

| Classifier | Task | Languages | Bundled |
|------------|------|-----------|---------|
| **Sentence-type** | question / command / statement / request / exclamation | 7 (en, de, es, fr, it, nl, pt) | EN bundled; others via HF |
| **EAT question-type** | 7 main / 53 fine-grained answer categories | EN | bundled |
| **Yes/No polarity** | yes / no / maybe (for statement answers) | 43 languages | multilingual bundled |

The package needs no internet connection on first use. English models ship inside the package. Other language models download automatically from HuggingFace on first use.

---

## Install

```bash
pip install little-questions
```

## Quick start

```python
from little_questions import Sentence
s = Sentence("Who invented the telephone?")
print(type(s).__name__)       # Question
print(s.sentence_type)        # question
print(s.classification)       # HUM:ind
print(s.main_label)           # HUM
print(s.secondary_label)      # ind
print(s.pretty_label)         # individual (Human)
print(f"{s.confidence:.2f}")  # 0.94
```

`Sentence(text)` dispatches to the right subclass automatically:

```python
from little_questions import Sentence, Question, Statement, Command, Request, Exclamation
s = Sentence("Who invented the telephone?")
assert isinstance(s, Question)
s = Sentence("Open the pod bay doors.")
assert isinstance(s, Command)
s = Sentence("Could you pass the salt?")
assert isinstance(s, Request)
assert isinstance(s, Command)   # Request subclasses Command
s = Sentence("The sky is blue.")
assert isinstance(s, Statement)
s = Sentence("What a beautiful day!")
assert isinstance(s, Exclamation)
```

---

## Sentence-type classification

Six classes: `wh_question`, `polar_question`, `statement`, `command`, `request`, `exclamation`.

`is_question` is `True` for both wh-questions and polar (yes/no) questions.

```python
s = Sentence("Is the Earth flat?")
print(s.sentence_type)  # polar_question
print(s.is_question)    # True
s = Sentence("What is the capital of France?")
print(s.sentence_type)  # wh_question
print(s.is_question)    # True
```

**Accuracy by language** (9,900 samples per language, held-out test set):

| Language | Accuracy | Macro F1 |
|----------|----------|----------|
| EN | 99.2% | 99.2% |
| NL | 98.8% | 98.8% |
| FR | 97.1% | 97.1% |
| IT | 97.0% | 97.0% |
| PT | 95.4% | 95.4% |
| DE | 85.6% | 84.9% |
| ES | 74.6% | 72.7% |

---

## EAT question-type classification (EN)

53 fine-grained labels across 7 main categories:

| Main | Meaning | Sub-types | Example |
|------|---------|-----------|---------|
| `ABBR` | Abbreviation | abb, exp | "What does NASA stand for?" |
| `BOOL` | Yes/No | yesno | "Is the Earth flat?" |
| `DESC` | Description | def, desc, manner, reason | "What is machine learning?" |
| `ENTY` | Entity | animal, body, color, food, ... | "What is the fastest land animal?" |
| `HUM` | Human | ind, gr, desc, title | "Who invented the telephone?" |
| `LOC` | Location | city, country, state, ... | "Where is the Eiffel Tower?" |
| `NUM` | Number / date | date, count, dist, money, ... | "When did World War II end?" |

Two-stage inference (eat7 gates eat53) achieves **93.4% macro F1**.

```python
s = Sentence("When did World War II end?")
print(s.classification)          # NUM:date
print(s.confidence)              # 0.97
# Full probability distribution over all 53 labels:
top3 = sorted(s.classification_scores.items(), key=lambda x: -x[1])[:3]
print(top3)  # [('NUM:date', 0.97), ('NUM:period', 0.01), ...]
```

### Unpunctuated / ASR input

For voice assistant output (no punctuation, lowercased), pass `punctuated=False`:

```python
s = Sentence("who invented the telephone", punctuated=False)
print(s.classification)   # HUM:ind
```

---

## Yes/No answer polarity

Detect whether a statement is an affirmative, negative, or uncertain answer. Works for 43 languages. The multilingual model ships bundled in the package.

```python
from little_questions import Sentence, Statement
a = Sentence("Yes, they can perceive some colors.")
assert isinstance(a, Statement)
print(a.answer_polarity)   # yes
print(a.is_affirmative)    # True
a = Sentence("No, dogs are colorblind.")
print(a.answer_polarity)   # no
print(a.is_negative)       # True
a = Sentence("It depends on the breed.")
print(a.answer_polarity)   # maybe
```

The `answer_polarity` property is lazy-loaded on first access.

**Per-language accuracy** (200 samples/language): 90-96% macro F1.
**Multilingual model** (bundled, all 43 languages): 84% macro F1.

---

## Multilingual usage

```python
s = Sentence("Qui a invente le telephone?", lang="fr")
print(s.sentence_type)    # question
a = Sentence("Oui, bien sur.", lang="fr")
print(a.answer_polarity)  # yes
```

---

## Routing pattern

```python
sentences = [
    "What is the capital of France?",
    "Turn off the lights.",
    "Could you help me?",
    "It is raining outside.",
    "How wonderful!",
]
for text in sentences:
    s = Sentence(text)
    if s.is_question:
        print(f"[QUESTION]     {text!r}  ->  {s.classification}")
    elif s.is_request:
        print(f"[REQUEST]      {text!r}")
    elif s.is_command:
        print(f"[COMMAND]      {text!r}")
    elif s.is_statement:
        print(f"[STATEMENT]    {text!r}")
    elif s.is_exclamation:
        print(f"[EXCLAMATION]  {text!r}")
```

---

## Model resolution

Models are resolved in order: **bundled -> user cache -> HuggingFace**.

| Location | Purpose |
|----------|---------|
| `little_questions/models/` (in-package) | Default offline models (EN + multilingual yesno) |
| `~/.local/share/little_questions/` | Downloaded or user-provided overrides |
| HuggingFace Hub | Auto-downloaded on first use for non-bundled models |

HuggingFace download requires:

```bash
pip install little-questions[hf]
```

---

## HuggingFace model repos

| Repo | Contents |
|------|---------|
| `TigreGotico/eat-classifiers` | EAT ONNX models (punctuated + unpunctuated) + benchmarks |
| `TigreGotico/sentence-types` | Sentence-type ONNX models (7 languages) |
| `TigreGotico/yes-no-classifiers` | Yes/No ONNX models (43 languages + multilingual) |

---

## Training your own models

```bash
pip install little-questions[train]
python -m train.train_eat                 # EAT question classifiers (EN)
python -m train.train_yesno               # yes/no polarity (43 langs + multilingual)
python -m train.train_sentence_type       # sentence-type classifiers
python -m train.benchmark_eat
python -m train.benchmark_sentence_type
python -m train.benchmark_yesno
python -m train.push_to_hf               # push all models to HuggingFace
```

See `docs/contributing.md` for the full development guide.

---

## License

Apache 2.0
