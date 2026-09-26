"""Training utilities for little_questions classifiers.

Standalone training module - install with: pip install little-questions[train]

Example:
    from train.classifiers import CalibratedLinearSVCClassifier

    clf = CalibratedLinearSVCClassifier()
    clf.train(train_data, labels)
    clf.save_onnx("eat53_svm_cal_EN_0.9.0.onnx")
"""

from __future__ import annotations

import re as _re
from os.path import join, dirname
from typing import List, Optional

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC as _LinearSVC
from sklearn.linear_model import LogisticRegression as _LogisticRegression
from sklearn.linear_model import SGDClassifier as _SGDClassifier

DATA_DIR = join(dirname(__file__), "clean_data")
REPORTS_DIR = join(dirname(__file__), "reports")


###################################################
# EAT categorical feature injection
###################################################
# Precision on EAT corpus (empirically verified):
#   __feat_yesno__   → BOOL  100% (is/does/do/was/are/were/has/will/did)
#   __feat_who__     → HUM    86%
#   __feat_where__   → LOC    99%
#   __feat_when__    → NUM    94%
#   __feat_why__     → DESC  100%
#   __feat_howmany__ → NUM    ~90% (how + quantifier)
#   __feat_define__  → DESC  100%

_PUNCT_RE = _re.compile(r"[^\w\s]", flags=_re.UNICODE)

_YESNO_STARTERS = frozenset({
    "is", "does", "do", "was", "are", "were", "has", "will", "did", "should",
})
_WHO_WORDS = frozenset({"who", "whom", "whose"})
_HOW_MANY = frozenset({"many", "much", "often", "long", "far", "old", "tall",
                        "deep", "wide", "fast", "big", "large", "heavy", "high"})
_DESC_VERBS = frozenset({"define", "explain", "meaning"})
_ABBR_WORDS = frozenset({"abbreviation", "acronym", "initials"})


def normalize_punctuated(text: str) -> str:
    """Lowercase — keeps punctuation (written/typed text)."""
    return text.lower().strip()


def normalize_unpunctuated(text: str) -> str:
    """Lowercase + strip all punctuation (ASR/spoken text)."""
    return _PUNCT_RE.sub("", text.lower()).strip()


def inject_eat_features(text: str) -> str:
    """Prepend __feat_*__ signal tokens to already-lowercased *text*.

    The ONNX model's TF-IDF vocabulary includes these unigrams, so they
    contribute like any other feature — no custom ONNX op required.
    """
    words = text.split()
    first = words[0] if words else ""
    second = words[1] if len(words) > 1 else ""
    tokens: list[str] = []

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
    if any(w in text for w in _DESC_VERBS):
        tokens.append("__feat_define__")
    if any(w in text for w in _ABBR_WORDS):
        tokens.append("__feat_abbr__")

    return (" ".join(tokens) + " " + text) if tokens else text


class EATTextPreprocessor(BaseEstimator, TransformerMixin):
    """Normalize + inject categorical tokens for use in sklearn Pipelines.

    punctuated=True (default): lowercase, keep punctuation — written/typed text.
    punctuated=False: lowercase, strip punctuation — ASR/spoken text.

    This transformer is stripped from the pipeline before ONNX export (see
    train_eat.save_onnx) and applied in Python at inference time instead.
    """

    def __init__(self, punctuated: bool = True) -> None:
        self.punctuated = punctuated

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        fn = normalize_punctuated if self.punctuated else normalize_unpunctuated
        return [inject_eat_features(fn(text)) for text in X]


###################################################
# Base classifier
###################################################

class TrainableClassifier:
    """Base class for trainable sklearn classifiers."""

    def __init__(self, pipeline_id: str) -> None:
        self.pipeline_id = pipeline_id.lower().split("-")[0]
        self.clf: Optional[Pipeline] = None

    @property
    def pipeline(self) -> list:
        return []

    def train(self, train_data: List[str], target_data: List[str]) -> Pipeline:
        self.clf = Pipeline(self.pipeline)
        self.clf.fit(train_data, target_data)
        return self.clf

    def predict(self, texts: List[str]) -> List[str]:
        if self.clf is None:
            raise RuntimeError("Classifier not trained. Call train() first.")
        return list(self.clf.predict(texts))

    def predict_proba(self, texts: List[str]) -> List[List[float]]:
        if self.clf is None:
            raise RuntimeError("Classifier not trained. Call train() first.")
        return self.clf.predict_proba(texts)

    def save_onnx(self, path: str) -> None:
        """Convert and save as ONNX model (uncalibrated LinearSVC)."""
        import json
        from pathlib import Path
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import StringTensorType
        import onnx
        from sklearn.svm import LinearSVC

        if self.clf is None:
            raise RuntimeError("Classifier not trained. Call train() first.")

        initial_type = [("input", StringTensorType([None]))]
        onnx_model = convert_sklearn(
            self.clf, initial_types=initial_type, options={LinearSVC: {"nocl": True}}
        )
        if isinstance(onnx_model, tuple):
            onnx_model = onnx_model[0]

        try:
            classes = list(list(self.clf.steps)[-1][1].classes_)
        except Exception:
            classes = []
        if classes:
            meta = onnx_model.metadata_props.add()
            meta.key = "classes"
            meta.value = json.dumps(classes)

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        onnx.save_model(onnx_model, path)


###################################################
# Classifier implementations
###################################################

class LinearSVCClassifier(TrainableClassifier):
    """Linear SVM text classifier (TF-IDF word n-grams)."""

    def __init__(self, pipeline_id: str = "linear-svc") -> None:
        super().__init__(pipeline_id)

    @property
    def pipeline(self) -> list:
        return [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.9, sublinear_tf=True)),
            ("clf", _LinearSVC(C=1.0, max_iter=2000)),
        ]


class CalibratedLinearSVCClassifier(TrainableClassifier):
    """Platt-calibrated Linear SVM (TF-IDF + EAT categorical token injection).

    Args:
        punctuated: True (default) for written/typed text (keeps punctuation);
                    False for ASR/spoken text (strips all punctuation).

    ONNX output[1] is a genuine probability vector (sum to 1.0).
    Export via save_onnx() which strips EATTextPreprocessor and embeds
    calibrated=true and punctuated metadata for inference-time preprocessing.
    """

    def __init__(self, pipeline_id: str = "calibrated-svc", punctuated: bool = True) -> None:
        super().__init__(pipeline_id)
        self.punctuated = punctuated

    @property
    def pipeline(self) -> list:
        tfidf = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True,
            max_features=20_000,
            token_pattern=r"(?u)\b\w+\b",  # includes __feat_*__ tokens
        )
        clf = CalibratedClassifierCV(_LinearSVC(C=1.0, max_iter=2000), cv=3, method="sigmoid")
        return [("prep", EATTextPreprocessor(punctuated=self.punctuated)), ("tfidf", tfidf), ("clf", clf)]

    def save_onnx(self, path: str) -> None:
        """Export to ONNX — strips EATTextPreprocessor, embeds calibrated + punctuated metadata."""
        import json
        from pathlib import Path
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import StringTensorType
        import onnx
        from sklearn.pipeline import Pipeline as _Pipeline

        if self.clf is None:
            raise RuntimeError("Classifier not trained. Call train() first.")

        exportable_steps = [(n, s) for n, s in self.clf.steps
                            if not isinstance(s, EATTextPreprocessor)]
        export_pipeline = _Pipeline(exportable_steps)

        initial_type = [("input", StringTensorType([None]))]
        options = {CalibratedClassifierCV: {"zipmap": False}}
        onnx_model = convert_sklearn(export_pipeline, initial_types=initial_type, options=options)
        if isinstance(onnx_model, tuple):
            onnx_model = onnx_model[0]

        try:
            classes = list(self.clf.steps[-1][1].classes_)
        except Exception:
            classes = []
        if classes:
            meta = onnx_model.metadata_props.add()
            meta.key = "classes"
            meta.value = json.dumps(classes)

        for key, val in [("calibrated", "true"),
                         ("punctuated", "true" if self.punctuated else "false")]:
            flag = onnx_model.metadata_props.add()
            flag.key = key
            flag.value = val

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        onnx.save_model(onnx_model, path)


class LogRegClassifier(TrainableClassifier):
    """Logistic Regression text classifier."""

    @property
    def pipeline(self) -> list:
        return [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.9, sublinear_tf=True)),
            ("clf", _LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0)),
        ]


class SGDClassifier(TrainableClassifier):
    """Stochastic Gradient Descent text classifier."""

    @property
    def pipeline(self) -> list:
        return [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.9, sublinear_tf=True)),
            ("clf", _SGDClassifier(loss="hinge", penalty="l2", alpha=1e-3, random_state=42)),
        ]


###################################################
# Model2Vec classifier
###################################################

class Model2VecClassifier:
    """Static embedding classifier using a model2vec Potion model + LinearSVC.

    Embeddings are produced by a frozen model2vec ``StaticModel``; a LinearSVC
    is trained on top.  Optionally fused with TF-IDF word n-grams for a small
    accuracy boost.

    Args:
        model_name: HuggingFace repo ID or local path of the model2vec model.
            Defaults to ``"minishlab/potion-base-8M"`` (EN-only).
            Use ``"minishlab/potion-multilingual-128M"`` for multilingual tasks.
        use_tfidf_fusion: When True, concatenate TF-IDF word(1,2) features with
            the model2vec embeddings before fitting the classifier.
    """

    name: str

    def __init__(
        self,
        model_name: str = "minishlab/potion-base-8M",
        use_tfidf_fusion: bool = False,
    ) -> None:
        self.model_name = model_name
        self.use_tfidf_fusion = use_tfidf_fusion
        self._m2v_model = None
        self._clf: Optional[_LinearSVC] = None
        self._tfidf: Optional[TfidfVectorizer] = None
        self._classes: Optional[List[str]] = None
        short = model_name.split("/")[-1]
        suffix = "+tfidf" if use_tfidf_fusion else ""
        self.name = f"m2v-{short}{suffix}"

    def _encode(self, texts: List[str]):
        return self._m2v_model.encode(texts)

    def _features(self, texts: List[str], fit: bool = False):
        import numpy as np
        from scipy.sparse import hstack

        emb = self._encode(texts)
        if not self.use_tfidf_fusion:
            return emb

        if fit:
            self._tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.4)
            tfidf_feat = self._tfidf.fit_transform(texts)
        else:
            tfidf_feat = self._tfidf.transform(texts)

        return hstack([tfidf_feat, emb])

    def train(self, train_data: List[str], target_data: List[str]) -> None:
        from model2vec import StaticModel

        self._m2v_model = StaticModel.from_pretrained(self.model_name)
        X = self._features(train_data, fit=True)
        self._clf = _LinearSVC()
        self._clf.fit(X, target_data)
        self._classes = list(self._clf.classes_)

    def predict(self, texts: List[str]) -> List[str]:
        if self._clf is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return list(self._clf.predict(self._features(texts)))

    def save(self, directory: str) -> None:
        import json
        import joblib
        from pathlib import Path

        if self._clf is None:
            raise RuntimeError("Model not trained. Call train() first.")

        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)

        joblib.dump(self._clf, out / "clf.joblib")
        if self.use_tfidf_fusion and self._tfidf is not None:
            joblib.dump(self._tfidf, out / "tfidf.joblib")

        self._m2v_model.save_pretrained(str(out / "model2vec"))

        meta = {
            "model_name": self.model_name,
            "use_tfidf_fusion": self.use_tfidf_fusion,
            "classes": self._classes,
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str) -> "Model2VecClassifier":
        import json
        import joblib
        from pathlib import Path
        from model2vec import StaticModel

        out = Path(directory)
        meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        inst = cls(
            model_name=meta["model_name"],
            use_tfidf_fusion=meta.get("use_tfidf_fusion", False),
        )
        inst._clf = joblib.load(out / "clf.joblib")
        inst._classes = meta.get("classes")
        inst._m2v_model = StaticModel.from_pretrained(str(out / "model2vec"))
        if inst.use_tfidf_fusion and (out / "tfidf.joblib").exists():
            inst._tfidf = joblib.load(out / "tfidf.joblib")
        return inst
