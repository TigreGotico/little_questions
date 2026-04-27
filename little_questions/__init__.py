from __future__ import annotations

from little_questions.constants import MAIN_LABEL_NAMES, SEC_LABEL_NAMES
from little_questions.classifiers import (
    EatClassifier,
    SentenceTypeClassifier,
    YesNoClassifier,
)


# ---------------------------------------------------------------------------
# Public helpers (used by tests + external code)
# ---------------------------------------------------------------------------

def get_classifier(lang: str = "en") -> EatClassifier:
    """Return the EatClassifier singleton for *lang*."""
    return EatClassifier.get_instance(lang)


def get_scorer(lang: str = "en") -> SentenceTypeClassifier:
    """Return the SentenceTypeClassifier singleton for *lang*."""
    return SentenceTypeClassifier.get_instance(lang)


def get_yesno_classifier(lang: str = "en") -> YesNoClassifier:
    """Return the YesNoClassifier singleton for *lang*."""
    return YesNoClassifier.get_instance(lang)


# ---------------------------------------------------------------------------
# Sentence and typed subclasses
# ---------------------------------------------------------------------------

class Sentence(str):
    """A classified sentence that subclasses :class:`str`.

    Attributes:
        classification: Full EAT label, e.g. ``"HUM:ind"``.
        classification_scores: Dict of EAT label → calibrated probability (sum ≈ 1.0).
        confidence: Max value from classification_scores.
        sentence_type: One of ``question``, ``command``, ``statement``,
            ``exclamation``, ``request``.
    """

    def __new__(cls, content: str, lang: str = "en") -> "Sentence":
        instance = str.__new__(cls, content)
        return instance

    def __init__(self, content: str, lang: str = "en") -> None:
        eat_clf = EatClassifier.get_instance(lang)
        sentence_clf = SentenceTypeClassifier.get_instance(lang)
        self.classification_scores: dict[str, float] = eat_clf.score(content)
        self.classification: str = max(
            self.classification_scores, key=self.classification_scores.__getitem__
        )
        self.confidence: float = self.classification_scores[self.classification]
        self.sentence_type: str = sentence_clf.predict(content)
        self.lang = lang

    @classmethod
    def parse(cls, text: str, lang: str = "en") -> "Sentence":
        """Alternative constructor — equivalent to ``Sentence(text, lang)``."""
        return cls(text, lang)

    @property
    def main_label(self) -> str:
        """Top-level EAT category (ABBR, BOOL, DESC, ENTY, HUM, LOC, NUM)."""
        return self.classification.split(":")[0]

    @property
    def secondary_label(self) -> str | None:
        """Fine-grained EAT subtype, or ``None`` when absent."""
        parts = self.classification.split(":")
        return parts[1] if len(parts) > 1 else None

    @property
    def pretty_label(self) -> str:
        """Human-readable label combining main and secondary EAT categories."""
        pretty_main = MAIN_LABEL_NAMES.get(self.main_label, self.main_label)
        sec = self.secondary_label
        if sec is None:
            return pretty_main
        pretty_sec = SEC_LABEL_NAMES.get(sec, sec)
        return f"{pretty_sec} ({pretty_main})"

    @property
    def is_exclamation(self) -> bool:
        return self.sentence_type == "exclamation"

    @property
    def is_request(self) -> bool:
        return self.sentence_type == "request"

    @property
    def is_statement(self) -> bool:
        return self.sentence_type == "statement"

    @property
    def is_command(self) -> bool:
        return self.sentence_type in ("command", "request")

    @property
    def is_question(self) -> bool:
        return self.sentence_type == "question"


class Question(Sentence):
    """A sentence classified as a question."""


class Statement(Sentence):
    """A sentence classified as a statement.

    When used as an answer to a yes/no question, the ``answer_polarity``
    property indicates whether the response is affirmative, negative, or
    neither (``"yes"``, ``"no"``, or ``"maybe"``).

    The classifier is loaded lazily on first access — no model download at
    construction time unless ``answer_polarity`` is actually called.
    """

    @property
    def answer_polarity(self) -> str:
        """``"yes"``, ``"no"``, or ``"maybe"``.

        Lazy-loads the yes/no ONNX classifier on first access.
        Raises ``RuntimeError`` if no model is available for this language.
        """
        if not hasattr(self, "_answer_polarity"):
            clf = YesNoClassifier.get_instance(self.lang)
            object.__setattr__(self, "_answer_polarity", clf.predict(str(self)))
        return self._answer_polarity

    @property
    def answer_polarity_scores(self) -> dict:
        """Calibrated probabilities over ``yes``/``no``/``maybe``.

        Lazy-loads the yes/no ONNX classifier on first access.
        Raises ``RuntimeError`` if no model is available for this language.
        """
        if not hasattr(self, "_answer_polarity_scores"):
            clf = YesNoClassifier.get_instance(self.lang)
            object.__setattr__(self, "_answer_polarity_scores", clf.score(str(self)))
        return self._answer_polarity_scores

    @property
    def is_affirmative(self) -> bool:
        return self.answer_polarity == "yes"

    @property
    def is_negative(self) -> bool:
        return self.answer_polarity == "no"


class Command(Sentence):
    """A sentence classified as a command."""


class Request(Command):
    """A sentence classified as a request (a polite command)."""


class Exclamation(Sentence):
    """A sentence classified as an exclamation."""


_SENTENCE_TYPE_TO_CLASS: dict[str, type] = {
    "question": Question,
    "statement": Statement,
    "command": Command,
    "request": Request,
    "exclamation": Exclamation,
}


def _sentence_factory(text: str, lang: str = "en") -> Sentence:
    """Create the correct Sentence subclass based on sentence_type."""
    # Temporarily create a plain Sentence to get the sentence_type
    s = Sentence.__new__(Sentence, text)
    Sentence.__init__(s, text, lang)
    cls = _SENTENCE_TYPE_TO_CLASS.get(s.sentence_type, Sentence)
    if cls is Sentence:
        return s
    # Re-instantiate as the correct subclass, copying already-computed attrs
    typed = str.__new__(cls, text)
    typed.classification = s.classification
    typed.classification_scores = s.classification_scores
    typed.confidence = s.confidence
    typed.sentence_type = s.sentence_type
    typed.lang = s.lang
    return typed


# Patch Sentence.__new__ so Sentence("...") returns the right subclass
_orig_sentence_new = Sentence.__new__


def _sentence_new(cls, content: str, lang: str = "en") -> "Sentence":
    if cls is Sentence:
        # defer to factory only when called as Sentence(...), not as subclass(...)
        return str.__new__(Sentence, content)
    return str.__new__(cls, content)


Sentence.__new__ = staticmethod(_sentence_new)  # type: ignore[assignment]


def _sentence_init(self, content: str, lang: str = "en") -> None:
    eat_clf = EatClassifier.get_instance(lang)
    sentence_clf = SentenceTypeClassifier.get_instance(lang)
    self.classification_scores = eat_clf.score(content)
    self.classification = max(
        self.classification_scores, key=self.classification_scores.__getitem__
    )
    self.confidence = self.classification_scores[self.classification]
    self.sentence_type = sentence_clf.predict(content)
    self.lang = lang
    # Reclassify into correct subclass if called as plain Sentence
    if type(self) is Sentence:
        target_cls = _SENTENCE_TYPE_TO_CLASS.get(self.sentence_type, Sentence)
        if target_cls is not Sentence:
            self.__class__ = target_cls


Sentence.__init__ = _sentence_init  # type: ignore[method-assign]
