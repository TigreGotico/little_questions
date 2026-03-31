"""Sentence type classifier using trained sklearn model.

Supports multiple languages: EN, ES, PT, CA, FR, DE, IT, NL
Labels: question, statement, command, exclamation, request

For other languages, falls back to rule-based heuristic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

from little_questions.constants import SENTENCE_TYPES, SUPPORTED_LANGUAGES


class SentenceTypeClassifier:
    """Trained sentence type classifier for multiple languages.

    Uses TF-IDF word(1,2) + char(3,4) with LinearSVC.
    Falls back to rule-based for unsupported languages.
    """

    _instances: Dict[str, "SentenceTypeClassifier"] = {}

    def __init__(self, lang: str = "en") -> None:
        self._lang = lang
        self._clf: Optional[Any] = None
        self._tfidf_word: Optional[Any] = None
        self._tfidf_char: Optional[Any] = None

    @classmethod
    def get_instance(cls, lang: str = "en") -> "SentenceTypeClassifier":
        """Get classifier instance for a language."""
        lang = lang.lower()
        if lang not in cls._instances:
            cls._instances[lang] = cls(lang)
            cls._instances[lang]._load()
        return cls._instances[lang]

    def _load(self) -> None:
        """Load model and vectorizers."""
        import joblib
        from scipy.sparse import hstack

        base_dir = Path(__file__).parent / "models"
        lang = self._lang

        clf_path = base_dir / f"sentence_type_svm_{lang}.joblib"
        if clf_path.exists():
            self._clf = joblib.load(clf_path)
            self._tfidf_word = joblib.load(
                base_dir / f"tfidf_word_sentence_{lang}.joblib"
            )
            self._tfidf_char = joblib.load(
                base_dir / f"tfidf_char_sentence_{lang}.joblib"
            )
        else:
            self._clf = None

    def predict(self, text: str) -> str:
        """Classify sentence type."""
        if self._clf is None:
            return self._fallback_predict(text)

        features = self._extract_features([text])
        return self._clf.predict(features)[0]

    def score(self, text: str) -> Dict[str, float]:
        """Return confidence scores for each sentence type."""
        if self._clf is None:
            return self._fallback_score(text)

        features = self._extract_features([text])
        decision = self._clf.decision_function(features)[0]

        exp_decision = np.exp(decision - np.max(decision))
        probs = exp_decision / exp_decision.sum()

        scores = {}
        for i, cls in enumerate(SENTENCE_TYPES):
            scores[cls] = float(probs[i])

        return scores

    def _extract_features(self, texts: List[str]) -> Any:
        """Extract combined TF-IDF features."""
        from scipy.sparse import hstack

        X_word = self._tfidf_word.transform(texts)
        X_char = self._tfidf_char.transform(texts)
        X = hstack([X_word, X_char])
        return X

    def _fallback_predict(self, text: str) -> str:
        """Fallback to rule-based prediction."""
        scores = self._fallback_score(text)
        return max(scores, key=lambda k: scores[k])

    def _fallback_score(self, text: str) -> Dict[str, float]:
        """Simple rule-based scoring."""
        text_lower = text.lower().strip()

        scores = {
            "question": 0.0,
            "statement": 0.0,
            "exclamation": 0.0,
            "command": 0.0,
            "request": 0.0,
        }

        if text.endswith("?"):
            scores["question"] = 0.8
        elif text.endswith("!"):
            scores["exclamation"] = 0.8
        elif text.endswith("."):
            scores["statement"] = 0.5
            scores["command"] = 0.5

        first_word = text_lower.split()[0] if text_lower.split() else ""

        if first_word in ["what", "why", "how", "when", "who", "which", "where"]:
            scores["question"] += 0.3
        elif first_word in ["would", "could", "can", "please"]:
            scores["request"] += 0.3
        elif first_word in ["let", "go", "come", "stop", "start"]:
            scores["command"] += 0.2

        return scores


def get_sentence_type_classifier(lang: str = "en") -> SentenceTypeClassifier:
    """Get the sentence type classifier instance for a language."""
    return SentenceTypeClassifier.get_instance(lang)
