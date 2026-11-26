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

Speech recognition output has no punctuation and no capital letters. `Sentence(text, punctuated=False)` selects the EAT models for this text. These models do not load: onnxruntime rejects their tokenizer pattern ([#78](https://github.com/TigreGotico/little_questions/issues/78)). Until #78 is fixed, give punctuated text only.

## Answer polarity

The yes/no classifier gives the polarity of an answer: `yes`, `no` or `maybe`. Call it directly:

```python
from little_questions import get_yesno_classifier
clf = get_yesno_classifier()
print(clf.predict("Yes, they can perceive some colors."))   # yes
print(clf.predict("No, dogs are colorblind."))              # no
print(clf.predict("It depends on the breed."))              # maybe
```

A `Statement` also has the `answer_polarity`, `is_affirmative` and `is_negative` properties. Most answers do not become a `Statement`: the sentence-type model classifies them as `Question` or `Command` ([#77](https://github.com/TigreGotico/little_questions/issues/77)). Until #77 is fixed, use `get_yesno_classifier()`.

## Other languages

`Sentence` works for English only. For each other language, `Sentence(text, lang=...)` raises `RuntimeError`, because it loads the EAT model first and only English EAT models exist ([#79](https://github.com/TigreGotico/little_questions/issues/79), [#85](https://github.com/TigreGotico/little_questions/issues/85)). Sentence-type models exist for de, es, fr, it, nl and pt, and they download from Hugging Face with the `hf` extra.

The yes/no classifier works for other languages:

```python
from little_questions import get_yesno_classifier
print(get_yesno_classifier().predict("Oui, bien sur."))  # yes
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
