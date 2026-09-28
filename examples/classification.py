"""Sentence classification examples — sentence type + EAT question answer type."""

from little_questions import Sentence, Question, Statement, Command, Request, Exclamation

# ---------------------------------------------------------------------------
# Automatic subclass dispatch
# ---------------------------------------------------------------------------

s = Sentence("Who invented the telephone?")
assert isinstance(s, Question)
assert s.sentence_type == "question"
assert s.classification == "HUM:ind"
assert s.main_label == "HUM"
assert s.secondary_label == "ind"
assert s.pretty_label == "individual (Human)"
print(f"Q: {s!r}  →  {s.classification}  ({s.pretty_label})  conf={s.confidence:.2f}")

s = Sentence("The sky is blue.")
assert isinstance(s, Statement)
assert s.sentence_type == "statement"
print(f"S: {s!r}  →  {s.sentence_type}")

s = Sentence("Open the pod bay doors.")
assert isinstance(s, Command)
assert s.is_command
print(f"C: {s!r}  →  {s.sentence_type}")

s = Sentence("Could you pass the salt?")
assert isinstance(s, Request)
assert isinstance(s, Command)   # Request subclasses Command
assert s.is_request
assert s.is_command
print(f"R: {s!r}  →  {s.sentence_type}")

s = Sentence("What a beautiful day!")
assert isinstance(s, Exclamation)
assert s.is_exclamation
print(f"E: {s!r}  →  {s.sentence_type}")

# ---------------------------------------------------------------------------
# Calibrated confidence scores
# ---------------------------------------------------------------------------

s = Sentence("When did World War II end?")
print(f"\nclassification:  {s.classification}")
print(f"confidence:      {s.confidence:.2f}")
top3 = sorted(s.classification_scores.items(), key=lambda x: -x[1])[:3]
print(f"top-3 labels:    {top3}")

# ---------------------------------------------------------------------------
# Routing pattern
# ---------------------------------------------------------------------------

sentences = [
    "What is the capital of France?",
    "Turn off the lights.",
    "Could you help me with this?",
    "It is raining outside.",
    "How fantastic!",
]

for text in sentences:
    s = Sentence(text)
    if s.is_question:
        print(f"[QUESTION]     {text!r}  →  {s.classification}")
    elif s.is_request:
        print(f"[REQUEST]      {text!r}")
    elif s.is_command:
        print(f"[COMMAND]      {text!r}")
    elif s.is_statement:
        print(f"[STATEMENT]    {text!r}")
    elif s.is_exclamation:
        print(f"[EXCLAMATION]  {text!r}")
