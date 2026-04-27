"""Sentence-type classifier — ONNX Runtime inference.

Supports multiple languages: EN, ES, PT, CA, FR, DE, IT, NL.
Labels: question, statement, command, exclamation, request.

Falls back to punctuation + first-word heuristics when no ONNX model is
available for the requested language.
"""
import abc
import json
import os
from abc import ABC
from threading import Lock
from typing import Dict, List, Optional, Union

import numpy as np
import onnxruntime as ort
from little_questions.constants import SENTENCE_TYPES, QUESTION_TYPES, SUPPORTED_LANGUAGES
from little_questions.models import get_sentence_type_model_path, get_model_path

_LOCALE_DIR = os.path.join(os.path.dirname(__file__), "locale")
_locale_cache: Dict[str, dict] = {}


def _load_locale(lang: str) -> dict:
    """Load locale JSON for *lang*, falling back to 'en' if not found.

    Resolves regional variants to their base 2-letter code
    (e.g. 'nl-BE', 'nl_BE', 'pt-BR' → 'nl', 'pt').
    """
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
    def score(self, text: str) -> Dict[str | int, float]:
        return NotImplemented


class HeuristicClassifier(Classifier):
    def __init__(self, lang: str):
        self.lang = lang

    def predict(self, text: str) -> str:
        scores = self._fallback_predict(text, self.lang)
        return max(scores, key=lambda k: scores[k])

    def score(self, text: str) -> Dict[str | int, float]:
        return self._fallback_predict(text, self.lang)

    @abc.abstractmethod
    def _fallback_predict(self, text: str, lang: str = "en") -> Dict[str, float]:
        return NotImplemented


class OnnxClassifier(Classifier):
    """ONNX-backed text classifier."""

    def __init__(self,
                 model_path: Optional[str] = None,
                 labels: Optional[List[str]] = None) -> None:
        self._session = None  # onnxruntime.InferenceSession, set by _load()
        self._classes: List[str] = labels or []
        self._load(model_path)

    def predict(self, text: str) -> str | int:
        """Return the most likely sentence type for *text*."""
        outputs = self._session.run(None, {"input": [text]})
        idx = int(outputs[0][0])
        return self._classes[idx] if self._classes else idx

    def score(self, text: str) -> Dict[str | int, float]:
        """Return softmax-normalised confidence scores for each sentence type."""
        outputs = self._session.run(None, {"input": [text]})
        # outputs[1]: raw decision-function scores, shape (1, n_classes)
        vals = np.array(outputs[1][0], dtype=np.float64)
        exp_v = np.exp(vals - vals.max())
        probs = exp_v / exp_v.sum()
        return {cls: float(p) for cls, p in zip(self._classes, probs)}

    def _load(self, model_path: str) -> None:
        """Load the ONNX model for this language.  Silently falls back to heuristic on error."""
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(model_path, opts)
        # Read class labels embedded as metadata at export time.
        meta = self._session.get_modelmeta().custom_metadata_map
        if "classes" in meta:
            self._classes = json.loads(meta["classes"])


###################################################
# Fallback heuristics in case model not available
###################################################
def _apply_rules(text: str, text_norm: str, words: List[str],
                 rules: List[dict], scores: Dict[str, float]) -> None:
    """Evaluate locale rules against *text_norm* and accumulate into *scores*.

    Positions:
      "end"        — text (original case) ends with the pattern
      "start"      — text_norm starts with the pattern
      "word_start" — first whitespace token of text_norm equals the pattern
      "anywhere"   — pattern is a substring of text_norm

    Rule fields:
      match            str or list[str]
      position         see above
      scores           {label: value}
      mode             "add" (default) or "max"
      suppress         zero out these labels after firing
      skip_if          skip rule if any of these labels already > 0
      exclusive_group  only the first firing rule per group applies
    """
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
            else:  # anywhere
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
# pretrained onnx classifiers
###################################################
class SentenceTypeClassifier(OnnxClassifier):
    """ONNX-backed sentence-type classifier."""
    _instances: Dict[str, Union["SentenceTypeClassifier", HeuristicSentenceTypeClassifier]] = {}
    _instances_lock: Lock = Lock()

    def __init__(self, model_path: str, labels: Optional[List[str]] = None) -> None:
        super().__init__(model_path, labels or list(SENTENCE_TYPES))

    @classmethod
    def get_instance(cls, lang: str) -> Union["SentenceTypeClassifier", HeuristicSentenceTypeClassifier]:
        """Return the cached classifier for *lang*, loading the ONNX model on first call."""
        lang = lang.lower()
        with cls._instances_lock:
            if lang not in cls._instances:
                if lang not in SUPPORTED_LANGUAGES:
                    cls._instances[lang] = HeuristicSentenceTypeClassifier(lang)
                else:
                    model_path = get_sentence_type_model_path(lang)
                    instance = cls(model_path, list(SENTENCE_TYPES))
                    cls._instances[lang] = instance
        return cls._instances[lang]


class QuestionTypeClassifier(OnnxClassifier):
    """ONNX-backed sentence-type classifier."""
    _instances: Dict[str, Union["QuestionTypeClassifier", HeuristicQuestionTypeClassifier]] = {}
    _instances_lock: Lock = Lock()

    def __init__(self, model_path: str, labels: Optional[List[str]] = None) -> None:
        super().__init__(model_path, labels or list(QUESTION_TYPES))

    @classmethod
    def get_instance(cls, lang: str) -> Union["QuestionTypeClassifier", HeuristicQuestionTypeClassifier]:
        """Return the cached classifier for *lang*, loading the ONNX model on first call."""
        lang = lang.lower()
        with cls._instances_lock:
            if lang not in cls._instances:
                if lang not in SUPPORTED_LANGUAGES:
                    cls._instances[lang] = HeuristicQuestionTypeClassifier(lang)
                else:
                    model_path = get_model_path(lang)
                    instance = cls(model_path, list(QUESTION_TYPES))
                    cls._instances[lang] = instance
        return cls._instances[lang]
