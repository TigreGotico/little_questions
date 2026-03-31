from __future__ import annotations

from typing import Optional

from little_questions.classifiers import get_classifier, get_scorer

_MAIN_LABEL_NAMES = {
    "HUM": "Human",
    "ENTY": "Entity",
    "DESC": "Description",
    "NUM": "Numeric",
    "LOC": "Location",
    "ABBR": "Abbreviation",
}

_SEC_LABEL_NAMES = {
    "def": "definition",
    "desc": "description",
    "ind": "individual",
    "dist": "distance",
    "volsize": "volume",
    "temp": "temperature",
    "gr": "group or organization of persons",
    "abb": "abbreviation",
    "exp": "expression abbreviated",
    "body": "organs of body",
    "cremat": "inventions, books and other creative pieces",
    "dismed": "diseases and medicine",
    "lang": "language",
    "termeq": "equivalent terms",
    "veh": "vehicles",
}

# str methods whose return value should NOT be re-wrapped as a Sentence subclass
_STR_PASSTHROUGH = frozenset(
    {
        "__len__",
        "__contains__",
        "__iter__",
        "__hash__",
        "__eq__",
        "__lt__",
        "__le__",
        "__gt__",
        "__ge__",
        "encode",
        "startswith",
        "endswith",
        "find",
        "rfind",
        "index",
        "rindex",
        "count",
        "isalpha",
        "isdigit",
        "isspace",
        "isalnum",
        "islower",
        "isupper",
        "istitle",
        "isidentifier",
        "isprintable",
        "isnumeric",
        "isdecimal",
    }
)


class Sentence(str):
    """A classified sentence that subclasses :class:`str`.

    The concrete subclass (``Question``, ``Command``, etc.) is determined at
    construction time by running the sentence scorer.  The COSC label is
    stored on :attr:`classification`.

    Attributes:
        classification: Full COSC label, e.g. ``"HUM:ind"``.
        model: Language/model identifier used for COSC classification.
        sentence_type: One of ``question``, ``command``, ``statement``,
            ``exclamation``, ``request``.
        score: Dict of sentence-type scores from the scorer.
    """

    # Populated by __new__; declared here so type checkers see the attributes.
    classification: str
    model: str
    sentence_type: str
    score: dict

    @classmethod
    def parse(cls, text: str, lang: str = "en") -> "Sentence":
        """Classify *text* in the given *lang* and return the appropriate ``Sentence`` subclass.

        This is the preferred entry point for parsing utterances. It is equivalent to
        calling ``Sentence(text, model=lang)`` but has a more discoverable signature.

        Args:
            text: The utterance to classify.
            lang: Language code (e.g. ``"en"``, ``"es"``, ``"fr"``). Defaults to ``"en"``.

        Returns:
            A concrete subclass instance (``Question``, ``Command``, ``Statement``,
            ``Exclamation``, or ``Request``) with classification metadata attached.
        """
        return cls(text, model=lang)

    def __new__(
        cls,
        content: str,
        model: str = "en",
        scorer: Optional[str] = None,
    ) -> "Sentence":
        """Classify *content* and return the appropriate ``Sentence`` subclass instance."""
        # lazy load classifiers
        question_classifier = get_classifier(model_id=model)
        sentence_classifier = get_scorer(lang=scorer or model)

        # classify
        classification = question_classifier.predict([content])[0]
        sent_classification = sentence_classifier.predict(content)

        # choose the correct concrete subclass
        _TYPE_MAP = {
            "command": Command,
            "question": Question,
            "exclamation": Exclamation,
            "statement": Statement,
            "request": Request,
        }
        target_cls = _TYPE_MAP.get(sent_classification, cls)
        obj = str.__new__(target_cls, content)

        # attach metadata
        obj.classification = classification
        obj.model = model
        obj.sentence_type = sent_classification
        obj.score = sentence_classifier.score(content)
        return obj

    def __getattr__(self, name: str):
        """Delegate unknown attribute access to :class:`str` (safety net only)."""
        # This is only reached when normal attribute lookup has already failed,
        # so it will not shadow real properties/methods defined on the class.
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    @property
    def main_label(self) -> str:
        """Top-level COSC category (HUM, ENTY, DESC, NUM, LOC, ABBR)."""
        return self.classification.split(":")[0]

    @property
    def secondary_label(self) -> str:
        """Fine-grained COSC subtype, e.g. ``ind``, ``def``, ``dist``."""
        return self.classification.split(":")[-1]

    @property
    def pretty_label(self) -> str:
        """Human-readable label combining main and secondary COSC categories."""
        pretty_main = _MAIN_LABEL_NAMES.get(self.main_label, self.main_label)
        pretty_sec = _SEC_LABEL_NAMES.get(self.secondary_label, self.secondary_label)
        return f"{pretty_sec} ({pretty_main})"

    @property
    def is_exclamation(self) -> bool:
        """``True`` when this sentence was classified as an exclamation."""
        return isinstance(self, Exclamation)

    @property
    def is_request(self) -> bool:
        """``True`` when this sentence was classified as a request."""
        return isinstance(self, Request)

    @property
    def is_statement(self) -> bool:
        """``True`` when this sentence was classified as a statement."""
        return isinstance(self, Statement)

    @property
    def is_command(self) -> bool:
        """``True`` when this sentence was classified as a command."""
        return isinstance(self, Command)

    @property
    def is_question(self) -> bool:
        """``True`` when this sentence was classified as a question."""
        return isinstance(self, Question)


class Question(Sentence):
    """A sentence classified as a question."""


class Command(Sentence):
    """A sentence classified as a command."""


class Request(Command):
    """A command phrased as a polite request."""


class Exclamation(Sentence):
    """A sentence classified as an exclamation."""


class Statement(Sentence):
    """A sentence classified as a declarative statement."""
