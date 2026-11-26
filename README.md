# little_questions
<!-- agent-hint: verify the docs with `pytest -q test/test_docs.py`; every number on a page comes from train/reports/. -->

little_questions classifies a sentence by type (question, command, request, statement, exclamation). For a question, it also gives the expected answer type (7 main and 53 fine-grained labels). For an answer, it gives the polarity (yes, no, maybe). All models are ONNX and run offline: the English models and the multilingual polarity model are in the package.

## Install

```bash
uv pip install little-questions
```

## Quick start

```python
from little_questions import Sentence
s = Sentence("Who invented the telephone?")
print(type(s).__name__)       # Question
print(s.classification)       # HUM:ind
print(s.pretty_label)         # individual (Human)
print(f"{s.confidence:.2f}")  # 0.88
```

## Documentation

| Page | Contents |
| --- | --- |
| [Index](docs/index.md) | Overview, headline result and reading order |
| [Usage](docs/usage.md) | Sentence types, answer types, polarity, other languages |
| [API](docs/api.md) | Classes, attributes and functions |
| [Models](docs/models.md) | Model files, where they load from, training |
| [Classification](docs/classification.md) | Benchmark results for each classifier |
| [Contributing](docs/contributing.md) | Development setup and tests |

## License

Apache 2.0
