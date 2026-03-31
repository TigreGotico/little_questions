"""Inference-only COSC question classifiers.

This module provides ONNX Runtime inference for COSC question classification.
Training pipelines have been moved to the `train/` module.
"""

from __future__ import annotations

from typing import List

from little_questions.models import get_model_path


from little_questions.classifiers.legacy import SentenceScorerHeuristic


class SentenceScorer:
    """Rule-based sentence-type scorer (language-agnostic fallback).

    Subclass and override the ``*_score`` static methods to implement
    language-specific heuristics.

    For the legacy POS-tagging based heuristics, use SentenceScorerHeuristic
    from little_questions.classifiers.legacy.
    """

    @staticmethod
    def predict(text: str) -> str:
        """Return the most likely sentence type for *text*."""
        score = SentenceScorer.score(text)
        best = max(score, key=lambda key: score[key])
        return best

    @staticmethod
    def score(text: str) -> dict:
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
        if text.endswith("?"):
            return 0.8
        return 0.4

    @staticmethod
    def statement_score(text: str) -> float:
        """Heuristic confidence that *text* is a statement."""
        if text.endswith("."):
            return 0.5
        return 0

    @staticmethod
    def exclamation_score(text: str) -> float:
        """Heuristic confidence that *text* is an exclamation."""
        if text.endswith("!"):
            return 0.6
        return 0

    @staticmethod
    def command_score(text: str) -> float:
        """Heuristic confidence that *text* is a command."""
        if text.endswith("."):
            return 0.6
        if text.endswith("!"):
            return 0.5
        return 0

    @staticmethod
    def request_score(text: str) -> float:
        """Heuristic confidence that *text* is a request."""
        if text.endswith("."):
            return 0.5
        if text.endswith("?"):
            return 0.5
        return 0


class Classifier:
    """COSC question classifier using ONNX Runtime inference.

    Supports ONNX models for cross-platform compatibility and performance.
    """

    def __init__(self, pipeline_id: str) -> None:
        self.pipeline_id = pipeline_id.lower().split("-")[0]
        self._onnx_session = None
        self._sklearn_clf = None
        self._classes = None

    @property
    def _is_onnx(self) -> bool:
        return self._onnx_session is not None

    def predict(self, texts: List[str]) -> List[str]:
        if self._onnx_session is not None:
            input_texts = [t for t in texts]  # 1D array of strings
            outputs = self._onnx_session.run(None, {"input": input_texts})
            labels = outputs[0]  # shape: (n_samples,)
            classes = self._classes
            if classes is None:
                classes = [f"class_{i}" for i in range(52)]  # fallback
            return [classes[int(l)] for l in labels]
        elif self._sklearn_clf is not None:
            return list(self._sklearn_clf.predict(texts))
        else:
            raise RuntimeError("Model not loaded. Call load_from_file() first.")

    def predict_proba(self, texts: List[str]) -> List[List[float]]:
        if self._sklearn_clf is not None:
            return self._sklearn_clf.predict_proba(texts)
        raise NotImplementedError("predict_proba not supported for ONNX models")

    def load_from_file(self, path: str = None) -> "Classifier":
        resolved_path = path or get_model_path(self.pipeline_id)

        if resolved_path.endswith(".onnx"):
            self._load_onnx(resolved_path)
        elif resolved_path.endswith(".pkl"):
            self._load_sklearn(resolved_path)
        else:
            try:
                self._load_onnx(resolved_path)
            except Exception:
                self._load_sklearn(resolved_path)
        return self

    def _load_onnx(self, path: str) -> None:
        """Load an ONNX model using onnxruntime."""
        import onnx
        import onnxruntime as ort

        onnx_model = onnx.load(path)
        self._classes = None

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self._onnx_session = ort.InferenceSession(path, sess_options)
        self._sklearn_clf = None

        # Try to get classes from sklearn fallback if available
        pkl_path = path.replace(".onnx", ".pkl")
        try:
            import joblib

            sklearn_clf = joblib.load(pkl_path)
            self._classes = list(sklearn_clf.classes_)
        except Exception:
            self._classes = None

    def _load_sklearn(self, path: str) -> None:
        import joblib

        self._sklearn_clf = joblib.load(path)
        self._onnx_session = None

        self._classes = None
        clf = self._sklearn_clf
        if clf is not None:
            classes = getattr(clf, "classes_", None)
            if classes is not None:
                self._classes = list(classes)


def get_scorer(lang=None):
    """Get the appropriate scorer for a language.

    Uses trained classifier for English (93% accuracy).
    Falls back to rule-based heuristic for other languages.
    """
    if lang:
        lang = lang.lower()
        if lang.startswith("en"):
            from little_questions.sentence_type import get_sentence_type_classifier

            return get_sentence_type_classifier()
    return SentenceScorer()


_LAZY_LOADING = {}


SUPPORTED_LANGUAGES = ["en", "es", "pt", "ca", "fr", "de", "it", "nl"]


def clear_classifier_cache() -> None:
    """Clear the global classifier cache."""
    global _LAZY_LOADING
    _LAZY_LOADING.clear()


def list_supported_languages():
    """Return the list of supported language codes."""
    return list(SUPPORTED_LANGUAGES)


def get_classifier(model_id: str) -> Classifier:
    """Load (or return cached) COSC classifier for the given language/model."""
    global _LAZY_LOADING
    model_id = model_id.lower()
    if model_id in _LAZY_LOADING:
        return _LAZY_LOADING[model_id]
    classifier = Classifier(model_id)
    classifier.load_from_file()
    _LAZY_LOADING[model_id] = classifier
    return classifier
