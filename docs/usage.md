# Usage
<!-- agent-hint: each example on this page runs in test/test_docs.py; a printed value must equal its comment. -->

```python
from little_questions import Sentence, Command, Request
s = Sentence("Could you pass the salt?")
print(type(s).__name__)            # Request
print(isinstance(s, Command))      # True
```

## Sentence types

`Sentence(text)` returns one of five subclasses: `Question`, `Statement`, `Command`, `Request` or `Exclamation`. `Request` is a subclass of `Command`.

The model has six classes: `wh_question`, `polar_question`, `statement`, `command`, `request` and `exclamation`. `sentence_type` gives `question` for the two question classes, and `is_question` is `True` for both.

```python
from little_questions import Sentence
s = Sentence("Is the Earth flat?")
print(s.sentence_type)  # question
print(s.is_question)    # True
s = Sentence("Open the pod bay doors.")
print(s.sentence_type)  # command
```

## Answer types of questions (English)

A question gets one of 53 labels in 7 main categories:

| Main | Meaning | Sub-types | Example |
| --- | --- | --- | --- |
| `ABBR` | Abbreviation | abb, exp | "What does NASA stand for?" |
| `BOOL` | Yes/No | yesno | "Is the Earth flat?" |
| `DESC` | Description | def, desc, manner, reason | "What is machine learning?" |
| `ENTY` | Entity | animal, body, color, food, ... | "What is the fastest land animal?" |
| `HUM` | Human | ind, gr, desc, title | "Who invented the telephone?" |
| `LOC` | Location | city, country, state, ... | "Where is the Eiffel Tower?" |
| `NUM` | Number or date | date, count, dist, money, ... | "When did World War II end?" |

```python
from little_questions import Sentence
s = Sentence("When did World War II end?")
print(s.classification)          # NUM:date
print(f"{s.confidence:.2f}")     # 0.99
top3 = sorted(s.classification_scores.items(), key=lambda x: -x[1])[:3]
print([(label, round(p, 2)) for label, p in top3])  # [('NUM:date', 0.99), ('NUM:period', 0.0), ('NUM:other', 0.0)]
```

### Text with no punctuation

Speech recognition output has no punctuation and no capital letters. For this text, set `punctuated=False`:

```python
from little_questions import Sentence
s = Sentence("who invented the telephone", punctuated=False)
print(s.classification)   # HUM:ind
```

## Answer polarity

A statement has `answer_polarity`: `yes`, `no` or `maybe`. The model loads on the first access.

```python
from little_questions import Sentence
a = Sentence("Yes, they can perceive some colors.")
print(a.answer_polarity)   # yes
print(a.is_affirmative)    # True
a = Sentence("No, dogs are colorblind.")
print(a.answer_polarity)   # no
print(a.is_negative)       # True
a = Sentence("It depends on the breed.")
print(a.answer_polarity)   # maybe
```

## Other languages

Give the language with `lang`. Sentence-type models exist for en, de, es, fr, it, nl and pt. The model for a language other than English downloads from Hugging Face on first use, which needs the `hf` extra.

```python
from little_questions import Sentence
s = Sentence("Qui a invente le telephone?", lang="fr")
print(s.sentence_type)    # question
a = Sentence("Oui, bien sur.", lang="fr")
print(a.answer_polarity)  # yes
```

## Route a sentence by its type

```python
from little_questions import Sentence
for text in ["What is the capital of France?", "Turn off the lights.", "Could you help me?"]:
    s = Sentence(text)
    if s.is_question:
        print("question", s.classification)
    elif s.is_request:
        print("request")
    elif s.is_command:
        print("command")
```

---
[Home](index.md) · [API →](api.md)
