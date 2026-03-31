"""Sentence type classifier using trained sklearn model.

Supports multiple languages: EN, ES, PT, CA, FR, DE, IT, NL
Labels: question, statement, command, exclamation, request

For other languages, falls back to rule-based heuristic.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from little_questions.constants import SENTENCE_TYPES, SUPPORTED_LANGUAGES

if TYPE_CHECKING:
    from sklearn.svm import LinearSVC
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.sparse import spmatrix


class SentenceTypeClassifier:
    """Trained sentence type classifier for multiple languages.

    Uses TF-IDF word(1,2) + char(3,4) with LinearSVC.
    Falls back to rule-based scoring when the model file is absent.
    """

    _instances: Dict[str, "SentenceTypeClassifier"] = {}
    _instances_lock: Lock = Lock()

    def __init__(self, lang: str = "en") -> None:
        self._lang = lang
        self._clf: Optional[LinearSVC] = None
        self._tfidf_word: Optional[TfidfVectorizer] = None
        self._tfidf_char: Optional[TfidfVectorizer] = None

    @classmethod
    def get_instance(cls, lang: str = "en") -> "SentenceTypeClassifier":
        """Return the (cached) classifier for *lang*, loading it on first call."""
        lang = lang.lower()
        with cls._instances_lock:
            if lang not in cls._instances:
                instance = cls(lang)
                instance._load()
                cls._instances[lang] = instance
        return cls._instances[lang]

    def _load(self) -> None:
        """Load model and vectorizers from bundled model files."""
        import joblib

        base_dir = Path(__file__).parent / "models"
        clf_path = base_dir / f"sentence_type_svm_{self._lang}.joblib"
        if clf_path.exists():
            self._clf = joblib.load(clf_path)
            self._tfidf_word = joblib.load(base_dir / f"tfidf_word_sentence_{self._lang}.joblib")
            self._tfidf_char = joblib.load(base_dir / f"tfidf_char_sentence_{self._lang}.joblib")
        else:
            self._clf = None

    def predict(self, text: str) -> str:
        """Return the most likely sentence type for *text*."""
        if self._clf is None:
            return self._fallback_predict(text)
        features = self._extract_features([text])
        return self._clf.predict(features)[0]

    def score(self, text: str) -> Dict[str, float]:
        """Return softmax-normalised confidence scores for each sentence type."""
        if self._clf is None:
            return self._fallback_score(text)

        features = self._extract_features([text])
        decision = self._clf.decision_function(features)[0]
        exp_d = np.exp(decision - np.max(decision))
        probs = exp_d / exp_d.sum()
        return {cls: float(probs[i]) for i, cls in enumerate(SENTENCE_TYPES)}

    def _extract_features(self, texts: List[str]) -> spmatrix:
        """Combine word-level and character-level TF-IDF features."""
        from scipy.sparse import hstack

        return hstack([
            self._tfidf_word.transform(texts),
            self._tfidf_char.transform(texts),
        ])

    def _fallback_predict(self, text: str) -> str:
        """Predict sentence type using punctuation heuristics."""
        scores = self._fallback_score(text)
        return max(scores, key=lambda k: scores[k])

    def _fallback_score(self, text: str) -> Dict[str, float]:
        """Simple rule-based scoring based on punctuation and first word."""
        text_lower = text.lower().strip()
        scores: Dict[str, float] = {t: 0.0 for t in SENTENCE_TYPES}

        if text.endswith("?"):
            scores["question"] = 0.8
        elif text.endswith("!"):
            scores["exclamation"] = 0.8
        elif text.endswith("."):
            scores["statement"] = 0.5
            scores["command"] = 0.5

        first_word = text_lower.split()[0] if text_lower else ""
        if first_word in {"what", "why", "how", "when", "who", "which", "where"}:
            scores["question"] += 0.3
        elif first_word in {"would", "could", "can", "please"}:
            scores["request"] += 0.3
        elif first_word in {"let", "go", "come", "stop", "start"}:
            scores["command"] += 0.2

        return scores


def get_sentence_type_classifier(lang: str = "en") -> SentenceTypeClassifier:
    """Return the sentence type classifier instance for *lang*."""
    return SentenceTypeClassifier.get_instance(lang)
