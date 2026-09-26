"""EAT question classification examples — all 7 main categories."""

from little_questions import Question, Sentence, Statement

questions = [
    # ABBR
    ("What does NASA stand for?",               "ABBR"),
    ("What is the abbreviation for kilometer?", "ABBR"),
    # BOOL
    ("Is the Earth flat?",                      "BOOL"),
    ("Can dogs see color?",                     "BOOL"),
    # DESC
    ("What is machine learning?",               "DESC"),
    ("Why is the sky blue?",                    "DESC"),
    ("How do vaccines work?",                   "DESC"),
    # ENTY
    ("What is the fastest land animal?",        "ENTY"),
    ("What color is the sun?",                  "ENTY"),
    # HUM
    ("Who invented the telephone?",             "HUM"),
    ("Who was the first person on the moon?",   "HUM"),
    # LOC
    ("Where is the Eiffel Tower?",              "LOC"),
    ("What country is Paris in?",               "LOC"),
    # NUM
    ("How old is the universe?",                "NUM"),
    ("When did World War II end?",              "NUM"),
    ("How fast is light?",                      "NUM"),
]

print(f"{'Question':<45}  {'Predicted':<12}  {'Expected':<12}  {'Conf':>6}")
print("-" * 80)
for text, expected_main in questions:
    q = Question(text)
    mark = "✓" if q.main_label == expected_main else "✗"
    print(f"{text:<45}  {q.classification:<12}  {expected_main:<12}  {q.confidence:>5.2f}  {mark}")

# ---------------------------------------------------------------------------
# Yes/no answer polarity (requires yesno ONNX model)
# ---------------------------------------------------------------------------

print("\n--- Yes/No answer polarity ---")
pairs = [
    ("Is water wet?",       "Yes, of course."),
    ("Can pigs fly?",       "No, they cannot."),
    ("Will it rain today?", "Maybe, it depends on the forecast."),
]

for question, answer in pairs:
    q = Sentence(question)
    a = Sentence(answer)
    assert isinstance(a, Statement)
    try:
        polarity = a.answer_polarity
        print(f"Q: {question!r}")
        print(f"A: {answer!r}  →  {polarity}  (is_affirmative={a.is_affirmative}, is_negative={a.is_negative})\n")
    except RuntimeError as e:
        print(f"(yes/no model not available: {e})")
        break
