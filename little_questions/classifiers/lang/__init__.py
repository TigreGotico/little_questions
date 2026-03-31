"""Language-specific sentence scorers.

This module provides trained sentence type classifiers.
The actual COSC classification is done via ONNX Runtime inference.
"""

from little_questions.sentence_type import (
    SentenceTypeClassifier,
    get_sentence_type_classifier,
)

__all__ = ["SentenceTypeClassifier", "get_sentence_type_classifier"]


def get_scorer(lang=None):
    """Get the appropriate scorer for a language."""
    if lang:
        lang = lang.lower()
        if lang.startswith("en"):
            return get_sentence_type_classifier("en")
    from little_questions.classifiers.base import SentenceScorer

    return SentenceScorer()
