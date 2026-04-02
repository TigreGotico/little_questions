"""Inference-only COSC question classifiers — ONNX Runtime only.

Training pipelines live in the ``train/`` module.
"""

from __future__ import annotations

from threading import Lock
from typing import List, Optional

from little_questions.constants import SUPPORTED_LANGUAGES
from little_questions.models import get_model_path


class Classifier:
    """COSC question classifier using ONNX Runtime inference.

    Attributes:
        pipeline_id: The language code this classifier was created for.
    """

    def __init__(self, pipeline_id: str) -> None:
        self.pipeline_id = pipeline_id.lower().split("-")[0]
        self._session = None   # onnxruntime.InferenceSession
        self._classes: Optional[List[str]] = None

    def predict(self, texts: List[str]) -> List[str]:
        """Classify *texts* and return one COSC label per entry.

        Args:
            texts: Input sentences.

        Returns:
            List of COSC label strings (e.g. ``["HUM:ind", "LOC:city"]``).

        Raises:
            RuntimeError: If the model has not been loaded yet.
        """
        if self._session is None:
            raise RuntimeError("Model not loaded. Call load_from_file() first.")
        outputs = self._session.run(None, {"input": texts})
        labels = outputs[0]
        classes = self._classes or [f"class_{i}" for i in range(52)]
        return [classes[int(label)] for label in labels]

    def load_from_file(self, path: Optional[str] = None) -> "Classifier":
        """Load an ONNX model from *path* (or the default model path for this language).

        Args:
            path: Absolute path to an ``.onnx`` file.  When omitted the path is
                  resolved via :func:`~little_questions.models.get_model_path`.

        Returns:
            *self*, for chaining.
        """
        import onnxruntime as ort

        resolved = path or get_model_path(self.pipeline_id)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(resolved, opts)

        # Read class labels: prefer ONNX model metadata, fall back to pkl sidecar.
        meta = self._session.get_modelmeta().custom_metadata_map
        if "classes" in meta:
            import json
            self._classes = json.loads(meta["classes"])
        else:
            # Sidecar .pkl contains the fitted pipeline; we only need .classes_.
            import joblib
            pkl_path = resolved.replace(".onnx", ".pkl")
            try:
                sidecar = joblib.load(pkl_path)
                self._classes = list(sidecar.classes_)
            except Exception:
                self._classes = None

        return self


_LAZY_LOADING: dict = {}
_LAZY_LOADING_LOCK: Lock = Lock()


def clear_classifier_cache() -> None:
    """Clear the global classifier cache (frees RAM; models reload on next use)."""
    with _LAZY_LOADING_LOCK:
        _LAZY_LOADING.clear()


def list_supported_languages() -> List[str]:
    """Return the list of supported language codes."""
    return list(SUPPORTED_LANGUAGES)


def get_classifier(model_id: str) -> Classifier:
    """Return the (cached) COSC classifier for *model_id*.

    Loads and caches the ONNX model on the first call per language.

    Args:
        model_id: Language code, e.g. ``"en"``, ``"es"``.

    Returns:
        Loaded :class:`Classifier` instance.
    """
    model_id = model_id.lower()
    with _LAZY_LOADING_LOCK:
        if model_id not in _LAZY_LOADING:
            _LAZY_LOADING[model_id] = Classifier(model_id).load_from_file()
    return _LAZY_LOADING[model_id]


def get_scorer(lang: Optional[str] = None):
    """Return the sentence-type classifier for *lang*.

    Uses a trained ONNX :class:`~little_questions.sentence_type.SentenceTypeClassifier`.
    Falls back to its internal heuristic when no model file is available.

    Args:
        lang: Language code.  Defaults to ``"en"``.

    Returns:
        :class:`~little_questions.sentence_type.SentenceTypeClassifier` instance.
    """
    from little_questions.sentence_type import get_sentence_type_classifier
    return get_sentence_type_classifier(lang or "en")
