# Little Questions

**Inference-only COSC question classifier for voice assistants.**

Classify sentences as questions, commands, statements, exclamations, or requests, and identify what type of answer is expected (COSC taxonomy: Human, Entity, Description, Numeric, Location, Abbreviation).

> **Status:** v0.8.0 - Fresh restart. ONNX models need to be retrained.

## Install

```bash
pip install little-questions
```

## Usage

```python
from little_questions import Question, Sentence

text = "who made you"
question = Sentence(text)

assert question.is_question
assert isinstance(question, Question)
assert question.pretty_label == "individual (Human)"
assert question.main_label == "HUM"
assert question.secondary_label == "ind"
```

### Sentence Types

- **Question** - Asks for information (e.g., "What is the capital of France?")
- **Command** - Directives (e.g., "Open the door")
- **Statement** - Declarative sentences (e.g., "The sky is blue")
- **Exclamation** - Expressive sentences (e.g., "What a day!")
- **Request** - Polite commands (e.g., "Could you pass the salt?")

### Supported Languages

- English (`en`)
- Spanish (`es`)
- Portuguese (`pt`)
- Catalan (`ca`)
- French (`fr`)
- German (`de`)
- Italian (`it`)
- Dutch (`nl`)

### COSC Taxonomy

Questions are classified into 6 main categories and 52 subtypes:

| Main | Subtypes |
|------|----------|
| HUM (Human) | individual, group, title, description |
| ENTY (Entity) | animal, body, color, food, product, vehicle, etc. |
| DESC (Description) | definition, description, manner, reason |
| NUM (Numeric) | date, distance, money, speed, temperature, etc. |
| LOC (Location) | city, country, mountain, state |
| ABBR (Abbreviation) | abbreviation, expansion |

## Architecture

```
┌─────────────────────────────────────────────┐
│  little_questions                          │
│  ├── Sentence.parse(text, lang)             │
│  ├── Sentence(text) → Question/Command/...  │
│  └── ONNX Runtime inference                │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  train/ (separate package)                  │
│  ├── classifiers.py - sklearn pipelines      │
│  ├── train_en.py - training scripts         │
│  └── clean_data/ - training datasets       │
└─────────────────────────────────────────────┘
```

## Training (Separate Package)

Training requires the `train` module:

```bash
pip install little-questions[train]
```

Train models:

```bash
python -m train.train_en          # 52-class model
python -m train.train_en --6     # 6-class model
python -m train.train_en --onnx  # Export as ONNX
```

Models are exported as ONNX format for cross-platform inference.

## Dependencies

**Runtime:**
- `onnxruntime` - ONNX model inference
- `brill_postagger` - POS tagging for sentence scoring
- `nltk`, `wordfreq` - NLP utilities

**Training (optional):**
- `scikit-learn` - Model training
- `skl2onnx` - ONNX export
- `onnx` - Model serialization
