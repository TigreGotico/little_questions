"""Punctuation-heuristic sentence-type scorer (language-agnostic fallback).

Used by get_scorer() for all non-English languages.  For English, the trained
SentenceTypeClassifier in little_questions.sentence_type is used instead.
"""

from __future__ import annotations

from typing import Dict


class SentenceScorer:
    """Rule-based sentence-type scorer (language-agnostic fallback).

    Subclass and override the ``*_score`` static methods to implement
    language-specific heuristics.
    """

    @staticmethod
    def predict(text: str) -> str:
        """Return the most likely sentence type for *text*."""
        scores = SentenceScorer.score(text)
        return max(scores, key=lambda key: scores[key])

    @staticmethod
    def score(text: str) -> Dict[str, float]:
        """Return a dict mapping sentence type to confidence score."""
        return {
            "question": SentenceScorer.question_score(text),
            "statement": SentenceScorer.statement_score(text),
            "exclamation": SentenceScorer.exclamation_score(text),
            "command": SentenceScorer.command_score(text),
            "request": SentenceScorer.request_score(text),
        }

    @staticmethod
    def question_score(text: str) -> float:
        """Heuristic confidence that *text* is a question."""
        return 0.8 if text.endswith("?") else 0.4

    @staticmethod
    def statement_score(text: str) -> float:
        """Heuristic confidence that *text* is a statement."""
        return 0.5 if text.endswith(".") else 0.0

    @staticmethod
    def exclamation_score(text: str) -> float:
        """Heuristic confidence that *text* is an exclamation."""
        return 0.6 if text.endswith("!") else 0.0

    @staticmethod
    def command_score(text: str) -> float:
        """Heuristic confidence that *text* is a command."""
        if text.endswith("."):
            return 0.6
        if text.endswith("!"):
            return 0.5
        return 0.0

    @staticmethod
    def request_score(text: str) -> float:
        """Heuristic confidence that *text* is a request."""
        if text.endswith(".") or text.endswith("?"):
            return 0.5
        return 0.0
