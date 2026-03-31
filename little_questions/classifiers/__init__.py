"""Inference-only COSC question classifiers.

This module provides ONNX Runtime inference for COSC question classification.
Training pipelines have been moved to the `train/` module.
"""

from __future__ import annotations

from typing import List

from little_questions.constants import SUPPORTED_LANGUAGES
from little_questions.models import get_model_path
from little_questions.classifiers.base import SentenceScorer
from little_questions.classifiers.legacy import SentenceScorerHeuristic


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
        """True when an ONNX model is loaded."""
        return self._onnx_session is not None

    def predict(self, texts: List[str]) -> List[str]:
        """Classify *texts* and return a label per entry."""
        if self._onnx_session is not None:
            input_texts = [t for t in texts]
            outputs = self._onnx_session.run(None, {"input": input_texts})
            labels = outputs[0]
            classes = self._classes or [f"class_{i}" for i in range(52)]
            return [classes[int(label)] for label in labels]
        elif self._sklearn_clf is not None:
            return list(self._sklearn_clf.predict(texts))
        else:
            raise RuntimeError("Model not loaded. Call load_from_file() first.")

    def predict_proba(self, texts: List[str]) -> List[List[float]]:
        """Return probability estimates for *texts* (sklearn only)."""
        if self._sklearn_clf is not None:
            return self._sklearn_clf.predict_proba(texts)
        raise NotImplementedError("predict_proba not supported for ONNX models")

    def load_from_file(self, path: str = None) -> "Classifier":
        """Load a model from *path* (or the default model path).

        Detects format from file extension: .onnx uses ONNX Runtime, .pkl uses joblib.
        """
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
        """Load an ONNX model using onnxruntime.

        Class labels are sourced from a companion .pkl sidecar if present,
        otherwise falls back to generic names.
        """
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._onnx_session = ort.InferenceSession(path, sess_options)
        self._sklearn_clf = None

        # Prefer class labels from a companion .pkl sidecar produced during training.
        pkl_path = path.replace(".onnx", ".pkl")
        try:
            import joblib
            sklearn_clf = joblib.load(pkl_path)
            self._classes = list(sklearn_clf.classes_)
        except Exception:
            self._classes = None

    def _load_sklearn(self, path: str) -> None:
        """Load a joblib-serialised sklearn pipeline."""
        import joblib

        self._sklearn_clf = joblib.load(path)
        self._onnx_session = None
        classes = getattr(self._sklearn_clf, "classes_", None)
        self._classes = list(classes) if classes is not None else None


_LAZY_LOADING: dict = {}


def clear_classifier_cache() -> None:
    """Clear the global classifier cache (frees RAM; models reload on next use)."""
    global _LAZY_LOADING
    _LAZY_LOADING.clear()


def list_supported_languages() -> List[str]:
    """Return the list of supported language codes."""
    return list(SUPPORTED_LANGUAGES)


def get_classifier(model_id: str) -> Classifier:
    """Load (or return cached) COSC classifier for the given language/model."""
    global _LAZY_LOADING
    model_id = model_id.lower()
    if model_id not in _LAZY_LOADING:
        classifier = Classifier(model_id)
        classifier.load_from_file()
        _LAZY_LOADING[model_id] = classifier
    return _LAZY_LOADING[model_id]


def get_scorer(lang: str = None):
    """Return the appropriate sentence-type scorer for *lang*.

    English uses the trained SentenceTypeClassifier (93% accuracy).
    All other languages fall back to the punctuation-heuristic SentenceScorer.
    """
    if lang and lang.lower().startswith("en"):
        from little_questions.sentence_type import get_sentence_type_classifier
        return get_sentence_type_classifier()
    return SentenceScorer()
