from little_questions.constants import MAIN_LABEL_NAMES, SEC_LABEL_NAMES
from little_questions.classifiers import EatClassifier, QuestionTypeClassifier, SentenceTypeClassifier


class Sentence(str):
    """A classified sentence that subclasses :class:`str`.

    The concrete subclass (``Question``, ``Command``, etc.) is determined at
    construction time by running the sentence scorer.  The EAT label is stored
    on :attr:`classification`.

    Attributes:
        classification: Full EAT label, e.g. ``"HUM:ind"``.
        classification_scores: Dict of EAT label → calibrated probability (sum ≈ 1.0).
        confidence: Max value from classification_scores (float in [0, 1]).
        sentence_type: One of ``question``, ``command``, ``statement``,
            ``exclamation``, ``request``.
    """

    def __init__(self, content: str, lang: str = "en") -> None:
        super().__init__(content)
        eat_clf = EatClassifier.get_instance(lang)
        sentence_classifier = SentenceTypeClassifier.get_instance(lang)
        self.classification_scores: dict[str, float] = eat_clf.score(content)
        self.classification: str = max(self.classification_scores, key=self.classification_scores.__getitem__)
        self.confidence: float = self.classification_scores[self.classification]
        self.sentence_type: str = sentence_classifier.predict(content)
        self.lang = lang

    @property
    def main_label(self) -> str:
        """Top-level EAT category (ABBR, BOOL, DESC, ENTY, HUM, LOC, NUM)."""
        return self.classification.split(":")[0]

    @property
    def secondary_label(self) -> str | None:
        """Fine-grained EAT subtype, e.g. ``ind``, ``def``, ``yesno``.

        Returns ``None`` for labels that have no subtype (not expected in EAT
        but handled defensively).
        """
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
        return self.sentence_type == "command"

    @property
    def is_question(self) -> bool:
        return self.sentence_type == "question"
