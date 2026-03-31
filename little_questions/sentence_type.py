"""Sentence-type classifier — ONNX Runtime inference.

Supports multiple languages: EN, ES, PT, CA, FR, DE, IT, NL.
Labels: question, statement, command, exclamation, request.

Falls back to punctuation + first-word heuristics when no ONNX model is
available for the requested language.
"""

from __future__ import annotations

from threading import Lock
from typing import Dict, List, Optional

import numpy as np

from little_questions.constants import SENTENCE_TYPES


class SentenceTypeClassifier:
    """ONNX-backed sentence-type classifier.

    Loads the ONNX model for *lang* on first use.  Falls back to a lightweight
    punctuation + first-word heuristic when the model file is unavailable.

    Attributes:
        lang: The language code this instance was created for.
    """

    _instances: Dict[str, "SentenceTypeClassifier"] = {}
    _instances_lock: Lock = Lock()

    def __init__(self, lang: str = "en") -> None:
        self.lang = lang
        self._session = None   # onnxruntime.InferenceSession, set by _load()
        self._classes: Optional[List[str]] = None

    @classmethod
    def get_instance(cls, lang: str = "en") -> "SentenceTypeClassifier":
        """Return the cached classifier for *lang*, loading the ONNX model on first call."""
        lang = lang.lower()
        with cls._instances_lock:
            if lang not in cls._instances:
                instance = cls(lang)
                instance._load()
                cls._instances[lang] = instance
        return cls._instances[lang]

    def _load(self) -> None:
        """Load the ONNX model for this language.  Silently falls back to heuristic on error."""
        import json
        import logging
        from little_questions.models import get_sentence_type_model_path

        path = get_sentence_type_model_path(self.lang)
        if path is None:
            return

        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(path, opts)

            # Read class labels embedded as metadata at export time.
            meta = self._session.get_modelmeta().custom_metadata_map
            if "classes" in meta:
                self._classes = json.loads(meta["classes"])
            else:
                self._classes = list(SENTENCE_TYPES)
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Failed to load sentence-type ONNX for %s: %s. Using heuristic fallback.",
                self.lang, exc,
            )
            self._session = None

    def predict(self, text: str) -> str:
        """Return the most likely sentence type for *text*."""
        if self._session is None:
            return self._fallback_predict(text)
        outputs = self._session.run(None, {"input": [text]})
        return self._classes[int(outputs[0][0])]

    def score(self, text: str) -> Dict[str, float]:
        """Return softmax-normalised confidence scores for each sentence type."""
        if self._session is None:
            return self._fallback_score(text)

        outputs = self._session.run(None, {"input": [text]})
        # outputs[1]: raw decision-function scores, shape (1, n_classes)
        vals = np.array(outputs[1][0], dtype=np.float64)
        exp_v = np.exp(vals - vals.max())
        probs = exp_v / exp_v.sum()
        return {cls: float(p) for cls, p in zip(self._classes, probs)}

    # ------------------------------------------------------------------
    # Heuristic fallback (no model required)
    # ------------------------------------------------------------------

    def _fallback_predict(self, text: str) -> str:
        scores = self._fallback_score(text)
        return max(scores, key=lambda k: scores[k])

    def _fallback_score(self, text: str) -> Dict[str, float]:
        """Lightweight heuristic based on punctuation and first word."""
        text_lower = text.lower().strip()
        scores: Dict[str, float] = {t: 0.0 for t in SENTENCE_TYPES}

        if text.endswith("?"):
            scores["question"] = 0.8
        elif text.endswith("!"):
            scores["exclamation"] = 0.8
        elif text.endswith("."):
            scores["statement"] = 0.5
            scores["command"] = 0.5

        first = text_lower.split()[0] if text_lower else ""
        if first in {"what", "why", "how", "when", "who", "which", "where"}:
            scores["question"] += 0.3
        elif first in {"would", "could", "can", "please"}:
            scores["request"] += 0.3
        elif first in {"let", "go", "come", "stop", "start"}:
            scores["command"] += 0.2

        return scores


def get_sentence_type_classifier(lang: str = "en") -> SentenceTypeClassifier:
    """Return the sentence-type classifier instance for *lang*."""
    return SentenceTypeClassifier.get_instance(lang)
