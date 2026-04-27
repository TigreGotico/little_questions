"""Classifiers for little_questions — ONNX Runtime inference only.

Models are downloaded automatically from HuggingFace on first use and cached
in ~/.local/share/little_questions/.
"""

from __future__ import annotations

import json
import os
import re as _re
from threading import Lock
from typing import Dict, List, Optional

import numpy as np
import onnxruntime as ort

from little_questions.constants import SENTENCE_TYPES, EAT_LABELS_53

###################################################
# EAT categorical feature injection
###################################################
# Synthetic token preprocessing applied at inference time to match how the
# svm_cal models were trained.  Tokens are injected BEFORE the ONNX session
# so the TF-IDF vocabulary (which includes __feat_*__ unigrams) picks them up.

_PUNCT_RE = _re.compile(r"[^\w\s]", flags=_re.UNICODE)

_YESNO_STARTERS = frozenset({"is", "does", "do", "was", "are", "were", "has", "will", "did", "should"})
_WHO_WORDS = frozenset({"who", "whom", "whose"})
_HOW_MANY = frozenset({"many", "much", "often", "long", "far", "old", "tall", "deep", "wide",
                        "fast", "big", "large", "heavy", "high"})
_DESC_VERBS = frozenset({"define", "explain", "meaning"})
_ABBR_WORDS = frozenset({"abbreviation", "acronym", "initials"})


def _eat_preprocess(text: str, punctuated: bool) -> str:
    """Normalise + inject __feat_*__ categorical tokens before ONNX inference."""
    low = (_PUNCT_RE.sub("", text.lower()) if not punctuated else text.lower()).strip()
    words = low.split()
    first = words[0] if words else ""
    second = words[1] if len(words) > 1 else ""
    tokens: List[str] = []
    if first in _YESNO_STARTERS:
        tokens.append("__feat_yesno__")
    if first in _WHO_WORDS:
        tokens.append("__feat_who__")
    if first == "where":
        tokens.append("__feat_where__")
    if first == "when":
        tokens.append("__feat_when__")
    if first == "why":
        tokens.append("__feat_why__")
    if first == "how":
        tokens.append("__feat_howmany__" if second in _HOW_MANY else "__feat_howother__")
    if any(w in low for w in _DESC_VERBS):
        tokens.append("__feat_define__")
    if any(w in low for w in _ABBR_WORDS):
        tokens.append("__feat_abbr__")
    return (" ".join(tokens) + " " + low) if tokens else low


###################################################
# ONNX session wrapper
###################################################

class _OnnxModel:
    """Thin wrapper around a single ONNX inference session.

    If the model metadata contains a ``punctuated`` key (set by the EAT training
    pipeline), preprocessing is applied in Python before each call.
    """

    def __init__(self, model_path: str) -> None:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(model_path, opts)
        meta = self._session.get_modelmeta().custom_metadata_map
        self.classes: List[str] = json.loads(meta.get("classes", "[]"))
        self.is_calibrated: bool = meta.get("calibrated") == "true"
        self._input_name: str = self._session.get_inputs()[0].name
        _p = meta.get("punctuated")
        self._punctuated: Optional[bool] = (True if _p == "true" else False) if _p is not None else None

    def _preprocess(self, text: str) -> str:
        if self._punctuated is None:
            return text
        return _eat_preprocess(text, self._punctuated)

    def _raw_scores(self, text: str) -> np.ndarray:
        inp = np.array([self._preprocess(text)], dtype=object)
        return np.array(self._session.run(None, {self._input_name: inp})[1][0], dtype=np.float64)

    def predict(self, text: str) -> str:
        inp = np.array([self._preprocess(text)], dtype=object)
        raw = self._session.run(None, {self._input_name: inp})[0][0]
        if isinstance(raw, (int, np.integer)):
            return self.classes[int(raw)]
        return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)

    def score(self, text: str) -> Dict[str, float]:
        vals = self._raw_scores(text)
        if not self.is_calibrated:
            exp_v = np.exp(vals - vals.max())
            vals = exp_v / exp_v.sum()
        return {cls: float(vals[i]) for i, cls in enumerate(self.classes)}


###################################################
# EAT question-type classifier (two-stage)
###################################################

class EatClassifier:
    """Two-stage calibrated EAT question-type classifier.

    Stage 1: eat7_svm_cal predicts the main category (ABBR/BOOL/DESC/ENTY/HUM/LOC/NUM).
    Stage 2: eat53_svm_cal scores all 53 labels; labels outside the stage-1 category
             are zeroed and the surviving probabilities are renormalised.

    Models are downloaded from TigreGotico/eat-classifiers on first use.
    Falls back to a single-stage 53-class model if the 7-class model is unavailable.
    Raises RuntimeError if no eat53 model can be fetched.
    """

    _instances: Dict[str, "EatClassifier"] = {}
    _instances_lock: Lock = Lock()
    _VERSION: str = "0.9.0"

    @classmethod
    def get_instance(cls, lang: str) -> "EatClassifier":
        lang = lang.lower()
        with cls._instances_lock:
            if lang not in cls._instances:
                cls._instances[lang] = cls._load(lang)
        return cls._instances[lang]

    @classmethod
    def _load(cls, lang: str) -> "EatClassifier":
        from little_questions.models import get_eat_model_path
        lang_upper = lang.upper()
        path53 = get_eat_model_path(f"eat53_svm_cal_{lang_upper}_{cls._VERSION}.onnx")
        path7 = get_eat_model_path(f"eat7_svm_cal_{lang_upper}_{cls._VERSION}.onnx")
        if path53 and os.path.isfile(path53):
            m53 = _OnnxModel(path53)
            m7 = _OnnxModel(path7) if path7 and os.path.isfile(path7) else None
            return cls(m53, m7)
        raise RuntimeError(
            f"No EAT ONNX model found for lang={lang!r}. "
            "Run `python -m train.train_eat` to train, or wait for models to be "
            "published to TigreGotico/eat-classifiers."
        )

    def __init__(self, model53: _OnnxModel, model7: Optional[_OnnxModel] = None) -> None:
        self._m53 = model53
        self._m7 = model7
        self._main_of_53: List[str] = [c.split(":")[0] for c in self._m53.classes]

    def predict(self, text: str) -> str:
        scores = self.score(text)
        return max(scores, key=scores.__getitem__)

    def score(self, text: str) -> Dict[str, float]:
        """Return renormalised calibrated probabilities over all 53 labels."""
        vals = self._m53._raw_scores(text)
        if not self._m53.is_calibrated:
            exp_v = np.exp(vals - vals.max())
            vals = exp_v / exp_v.sum()

        if self._m7 is not None:
            main = self._m7.predict(text)
            masked = np.where([m == main for m in self._main_of_53], vals, 0.0)
            total = masked.sum()
            if total > 0:
                vals = masked / total

        return {cls: float(vals[i]) for i, cls in enumerate(self._m53.classes)}


###################################################
# Sentence-type classifier
###################################################

class SentenceTypeClassifier:
    """ONNX-backed sentence-type classifier with HF auto-download.

    Raises RuntimeError if no model is available for the requested language.
    """

    _instances: Dict[str, "SentenceTypeClassifier"] = {}
    _instances_lock: Lock = Lock()

    @classmethod
    def get_instance(cls, lang: str) -> "SentenceTypeClassifier":
        lang = lang.lower()
        with cls._instances_lock:
            if lang not in cls._instances:
                cls._instances[lang] = cls._load(lang)
        return cls._instances[lang]

    @classmethod
    def _load(cls, lang: str) -> "SentenceTypeClassifier":
        from little_questions.models import get_sentence_type_model_path
        path = get_sentence_type_model_path(lang)
        if path and os.path.isfile(path):
            return cls(path, list(SENTENCE_TYPES))
        raise RuntimeError(
            f"No sentence-type ONNX model found for lang={lang!r}. "
            "Supported languages: en, de, es, fr, it, nl, pt."
        )

    def __init__(self, model_path: str, labels: List[str]) -> None:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(model_path, opts)
        meta = self._session.get_modelmeta().custom_metadata_map
        self._classes: List[str] = json.loads(meta.get("classes", "[]")) or labels
        self._input_name: str = self._session.get_inputs()[0].name

    # Older models output polar_question/wh_question instead of question
    _QUESTION_VARIANTS = frozenset({"polar_question", "wh_question", "question"})

    @staticmethod
    def _normalise(label: str) -> str:
        if label in SentenceTypeClassifier._QUESTION_VARIANTS:
            return "question"
        return label

    def predict(self, text: str) -> str:
        outputs = self._session.run(None, {self._input_name: np.array([text], dtype=object)})
        raw = outputs[0][0]
        if isinstance(raw, (int, np.integer)):
            raw = self._classes[int(raw)]
        elif isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return self._normalise(str(raw))

    def score(self, text: str) -> Dict[str, float]:
        outputs = self._session.run(None, {self._input_name: np.array([text], dtype=object)})
        vals = np.array(outputs[1][0], dtype=np.float64)
        exp_v = np.exp(vals - vals.max())
        probs = exp_v / exp_v.sum()
        merged: Dict[str, float] = {}
        for cls, p in zip(self._classes, probs):
            key = self._normalise(cls)
            merged[key] = merged.get(key, 0.0) + float(p)
        return merged


###################################################
# Yes/No answer polarity classifier (ONNX only)
###################################################

class YesNoClassifier:
    """Yes/No answer polarity classifier.

    Labels: ``yes``, ``no``, ``maybe``.

    Models are fetched from TigreGotico/yes-no-classifiers on first use
    (language-specific first, then multilingual fallback) and cached at
    ~/.local/share/little_questions/yesno/.

    Raises RuntimeError if no model can be loaded.
    """

    _instances: Dict[str, "YesNoClassifier"] = {}
    _instances_lock: Lock = Lock()
    _VERSION: str = "0.9.0"

    @classmethod
    def get_instance(cls, lang: str) -> "YesNoClassifier":
        lang = lang.lower()
        with cls._instances_lock:
            if lang not in cls._instances:
                cls._instances[lang] = cls._load(lang)
        return cls._instances[lang]

    @classmethod
    def _load(cls, lang: str) -> "YesNoClassifier":
        from little_questions.models import get_yesno_model_path
        path = get_yesno_model_path(lang, version=cls._VERSION)
        if path and os.path.isfile(path):
            return cls(path)
        raise RuntimeError(
            f"No yes/no ONNX model found for lang={lang!r}. "
            "Run `python -m train.train_yesno` to train and export the model, "
            "or wait for the model to be published to TigreGotico/yes-no-classifiers."
        )

    def __init__(self, model_path: str) -> None:
        self._model = _OnnxModel(model_path)

    def predict(self, text: str) -> str:
        return self._model.predict(text)

    def score(self, text: str) -> Dict[str, float]:
        return self._model.score(text)


###################################################
# Cache management
###################################################

def clear_classifier_cache() -> None:
    """Clear all cached classifier instances, forcing reload on next use."""
    EatClassifier._instances.clear()
    SentenceTypeClassifier._instances.clear()
    YesNoClassifier._instances.clear()


def list_supported_languages() -> List[str]:
    """Return languages that have a sentence-type ONNX model available."""
    from little_questions.models import SENTENCE_TYPE_ONNX
    return list(SENTENCE_TYPE_ONNX.keys())
