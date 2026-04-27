"""Classifiers for little_questions — ONNX Runtime inference.

Models are downloaded automatically from HuggingFace on first use and cached
in ~/.local/share/little_questions/.
"""

from __future__ import annotations

import abc
import json
import os
from abc import ABC
from threading import Lock
from typing import Dict, List, Optional, Union

import numpy as np
import onnxruntime as ort

from little_questions.constants import (
    SENTENCE_TYPES, EAT_LABELS_53, EAT_LABELS_7, SUPPORTED_LANGUAGES,
)

# Backward-compat alias used by heuristic classifier
QUESTION_TYPES = EAT_LABELS_53

###################################################
# resource file handling
###################################################
_LOCALE_DIR = os.path.join(os.path.dirname(__file__), "locale")
_locale_cache: Dict[str, dict] = {}


def _load_locale(lang: str) -> dict:
    """Load locale JSON for *lang*, falling back to 'en' if not found."""
    base = lang.lower().replace("_", "-").split("-")[0]
    if base not in _locale_cache:
        path = os.path.join(_LOCALE_DIR, f"{base}.json")
        if not os.path.isfile(path):
            path = os.path.join(_LOCALE_DIR, "en.json")
        with open(path, encoding="utf-8") as f:
            _locale_cache[base] = json.load(f)
    return _locale_cache[base]


###################################################
# base classes
###################################################
class Classifier(ABC):
    @abc.abstractmethod
    def predict(self, text: str) -> str:
        return NotImplemented

    @abc.abstractmethod
    def score(self, text: str) -> Dict[str, float]:
        return NotImplemented


class HeuristicClassifier(Classifier):
    def __init__(self, lang: str):
        self.lang = lang

    def predict(self, text: str) -> str:
        scores = self._fallback_predict(text, self.lang)
        return max(scores, key=lambda k: scores[k])

    def score(self, text: str) -> Dict[str, float]:
        return self._fallback_predict(text, self.lang)

    @abc.abstractmethod
    def _fallback_predict(self, text: str, lang: str = "en") -> Dict[str, float]:
        return NotImplemented


class OnnxClassifier(Classifier):
    """ONNX-backed text classifier."""

    def __init__(self,
                 model_path: Optional[str] = None,
                 labels: Optional[List[str]] = None) -> None:
        self._session = None
        self._classes: List[str] = labels or []
        self._load(model_path)

    def predict(self, text: str) -> str:
        outputs = self._session.run(None, {"input": [text]})
        idx = int(outputs[0][0])
        return self._classes[idx] if self._classes else idx

    def score(self, text: str) -> Dict[str, float]:
        outputs = self._session.run(None, {"input": [text]})
        vals = np.array(outputs[1][0], dtype=np.float64)
        exp_v = np.exp(vals - vals.max())
        probs = exp_v / exp_v.sum()
        return {cls: float(p) for cls, p in zip(self._classes, probs)}

    def _load(self, model_path: str) -> None:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(model_path, opts)
        meta = self._session.get_modelmeta().custom_metadata_map
        if "classes" in meta:
            self._classes = json.loads(meta["classes"])


###################################################
# Heuristic fallbacks
###################################################
def _apply_rules(text: str, text_norm: str, words: List[str],
                 rules: List[dict], scores: Dict[str, float]) -> None:
    first = words[0] if words else ""
    exclusive_fired: set = set()

    for rule in rules:
        excl = rule.get("exclusive_group")
        if excl and excl in exclusive_fired:
            continue
        if any(scores.get(lbl, 0.0) > 0.0 for lbl in rule.get("skip_if", [])):
            continue

        position = rule.get("position", "anywhere")
        patterns = rule["match"] if isinstance(rule["match"], list) else [rule["match"]]

        fired = False
        for pat in patterns:
            if position == "end":
                fired = text.endswith(pat)
            elif position == "start":
                fired = text_norm.startswith(pat)
            elif position == "word_start":
                fired = (first == pat)
            else:
                fired = (pat in text_norm)
            if fired:
                break

        if not fired:
            continue

        if excl:
            exclusive_fired.add(excl)

        mode = rule.get("mode", "add")
        for label, val in rule.get("scores", {}).items():
            if mode == "max":
                scores[label] = max(scores.get(label, 0.0), val)
            else:
                scores[label] = scores.get(label, 0.0) + val

        for label in rule.get("suppress", []):
            scores[label] = 0.0


class HeuristicSentenceTypeClassifier(HeuristicClassifier):

    def _fallback_predict(self, text: str, lang: str = "en") -> Dict[str, float]:
        locale = _load_locale(lang).get("sentence_type", {})
        strip = locale.get("strip_leading", "")
        text_norm = text.lower().strip()
        if strip:
            text_norm = text_norm.lstrip(strip)
        words = text_norm.split()
        scores: Dict[str, float] = {t: 0.0 for t in SENTENCE_TYPES}

        _apply_rules(text, text_norm, words, locale.get("rules", []), scores)

        if all(v == 0.0 for v in scores.values()):
            for label, val in locale.get("default", {"statement": 0.4}).items():
                scores[label] = val

        return scores


class HeuristicQuestionTypeClassifier(HeuristicClassifier):

    def _fallback_predict(self, text: str, lang: str = "en") -> Dict[str, float]:
        locale = _load_locale(lang).get("question_type", {})
        strip = locale.get("strip_leading", "")
        text_norm = text.lower().strip()
        if strip:
            text_norm = text_norm.lstrip(strip)
        words = text_norm.split()
        scores: Dict[str, float] = {t: 0.0 for t in QUESTION_TYPES}

        _apply_rules(text, text_norm, words, locale.get("rules", []), scores)

        threshold = locale.get("default_if_low_threshold", 0.3)
        if max(scores.values(), default=0.0) < threshold:
            for label, val in locale.get("default_if_low", {}).items():
                scores[label] = scores.get(label, 0.0) + val

        return scores


###################################################
# ONNX-backed singletons (with HF auto-download)
###################################################

class _OnnxModel:
    """Thin wrapper around a single ONNX inference session."""

    def __init__(self, model_path: str) -> None:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(model_path, opts)
        meta = self._session.get_modelmeta().custom_metadata_map
        self.classes: List[str] = json.loads(meta.get("classes", "[]"))
        self.is_calibrated: bool = meta.get("calibrated") == "true"
        self._input_name: str = self._session.get_inputs()[0].name

    def _raw_scores(self, text: str) -> np.ndarray:
        inp = np.array([text], dtype=object)
        return np.array(self._session.run(None, {self._input_name: inp})[1][0], dtype=np.float64)

    def predict(self, text: str) -> str:
        inp = np.array([text], dtype=object)
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


class EatClassifier:
    """Two-stage calibrated EAT question-type classifier (default) or single-stage fallback.

    Stage 1: eat7_svm_cal predicts the main category (ABBR/BOOL/DESC/ENTY/HUM/LOC/NUM).
    Stage 2: eat53_svm_cal scores all 53 labels; labels outside the stage-1 category
             are zeroed and the surviving probabilities are renormalised.

    Both models are downloaded from TigreGotico/eat-classifiers on first use.
    Falls back to a single-stage 53-class model if the 7-class model is unavailable,
    and to the heuristic classifier when neither can be fetched.
    """

    _instances: Dict[str, Union["EatClassifier", HeuristicQuestionTypeClassifier]] = {}
    _instances_lock: Lock = Lock()
    _VERSION: str = "0.9.0"

    @classmethod
    def get_instance(cls, lang: str) -> Union["EatClassifier", HeuristicQuestionTypeClassifier]:
        lang = lang.lower()
        with cls._instances_lock:
            if lang not in cls._instances:
                cls._instances[lang] = cls._load(lang)
        return cls._instances[lang]

    @classmethod
    def _load(cls, lang: str) -> Union["EatClassifier", HeuristicQuestionTypeClassifier]:
        from little_questions.models import get_eat_model_path
        lang_upper = lang.upper()
        path53 = get_eat_model_path(f"eat53_svm_cal_{lang_upper}_{cls._VERSION}.onnx")
        path7  = get_eat_model_path(f"eat7_svm_cal_{lang_upper}_{cls._VERSION}.onnx")
        if path53 and os.path.isfile(path53):
            m53 = _OnnxModel(path53)
            m7  = _OnnxModel(path7) if path7 and os.path.isfile(path7) else None
            return cls(m53, m7)
        return HeuristicQuestionTypeClassifier(lang)

    def __init__(self, model53: _OnnxModel, model7: Optional[_OnnxModel] = None) -> None:
        self._m53 = model53
        self._m7  = model7
        # Precompute which 53-class index belongs to which 7-class main category
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
            # if stage-1 is wrong and total==0, keep original probs as fallback

        return {cls: float(vals[i]) for i, cls in enumerate(self._m53.classes)}


# Backward-compat alias
QuestionTypeClassifier = EatClassifier


class SentenceTypeClassifier:
    """ONNX-backed sentence-type classifier with HF auto-download."""

    _instances: Dict[str, Union["SentenceTypeClassifier", HeuristicSentenceTypeClassifier]] = {}
    _instances_lock: Lock = Lock()

    @classmethod
    def get_instance(cls, lang: str) -> Union["SentenceTypeClassifier", HeuristicSentenceTypeClassifier]:
        lang = lang.lower()
        with cls._instances_lock:
            if lang not in cls._instances:
                cls._instances[lang] = cls._load(lang)
        return cls._instances[lang]

    @classmethod
    def _load(cls, lang: str) -> Union["SentenceTypeClassifier", HeuristicSentenceTypeClassifier]:
        from little_questions.models import get_sentence_type_model_path
        path = get_sentence_type_model_path(lang)
        if path and os.path.isfile(path):
            return cls(path, list(SENTENCE_TYPES))
        return HeuristicSentenceTypeClassifier(lang)

    def __init__(self, model_path: str, labels: List[str]) -> None:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(model_path, opts)
        meta = self._session.get_modelmeta().custom_metadata_map
        self._classes: List[str] = json.loads(meta.get("classes", "[]")) or labels
        self._input_name: str = self._session.get_inputs()[0].name

    # Older sentence-type models distinguish polar vs wh questions; normalise both.
    _QUESTION_VARIANTS = frozenset({"polar_question", "wh_question", "question"})

    @staticmethod
    def _normalise(label: str) -> str:
        """Map model-specific label variants to canonical SENTENCE_TYPES."""
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
        # Merge polar_question + wh_question into "question"
        merged: Dict[str, float] = {}
        for cls, p in zip(self._classes, probs):
            key = self._normalise(cls)
            merged[key] = merged.get(key, 0.0) + float(p)
        return merged


###################################################
# Cache management utilities
###################################################

# Exposed so tests can inspect / clear cached instances
_LAZY_LOADING: Dict[str, object] = {}


def clear_classifier_cache() -> None:
    """Clear all cached classifier instances, forcing reload on next use."""
    EatClassifier._instances.clear()
    SentenceTypeClassifier._instances.clear()
    _LAZY_LOADING.clear()


def list_supported_languages() -> List[str]:
    """Return languages that have a sentence-type ONNX model available."""
    from little_questions.models import SENTENCE_TYPE_ONNX
    return list(SENTENCE_TYPE_ONNX.keys())
