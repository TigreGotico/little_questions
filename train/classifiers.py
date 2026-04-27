"""Training utilities for little_questions classifiers.

Standalone training module - install with: pip install little-questions[train]

Example:
    from train.classifiers import LinearSVCClassifier

    clf = LinearSVCClassifier("en")
    clf.train(train_data, labels)
    clf.save("model.onnx")
"""

from __future__ import annotations

import os
from os.path import join, dirname
from typing import List, Optional

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC as _LinearSVC
from sklearn.linear_model import LogisticRegression as _LogisticRegression
from sklearn.naive_bayes import MultinomialNB as _MultinomialNB
from sklearn.ensemble import RandomForestClassifier as _RandomForestClassifier
from sklearn.linear_model import (
    PassiveAggressiveClassifier as _PassiveAggressiveClassifier,
)
from sklearn.linear_model import SGDClassifier as _SGDClassifier
from sklearn.linear_model import Perceptron as _Perceptron

DATA_DIR = join(dirname(__file__), "clean_data")
REPORTS_DIR = join(dirname(__file__), "reports")


def normalize_text(texts, stemmer=None):
    """Normalize text with optional stemming/lemmatization."""
    from nltk.stem import WordNetLemmatizer
    from nltk import word_tokenize

    if stemmer is None:
        stemmer = WordNetLemmatizer()

    normalized = []
    for text in texts:
        tokens = word_tokenize(text.lower())
        stemmed = [stemmer.lemmatize(token) for token in tokens]
        normalized.append(" ".join(stemmed))
    return normalized


class WordFeaturesVectorizer(BaseEstimator, TransformerMixin):
    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        return normalize_text(X, **transform_params)


# ---------------------------------------------------------------------------
# Categorical features via synthetic token injection
# ---------------------------------------------------------------------------
# Prepending __feat_*__ tokens to the question text before TF-IDF vectorization
# injects categorical signal without breaking skl2onnx export — the pipeline
# stays as pure sklearn string ops that skl2onnx handles natively.
#
# Precision on EAT corpus (empirically verified):
#   __feat_yesno__  → BOOL  100% precision (is/does/do/was/are/were/has/will/did)
#   __feat_who__    → HUM    86% precision
#   __feat_where__  → LOC    99% precision
#   __feat_when__   → NUM    94% precision
#   __feat_why__    → DESC  100% precision
#   __feat_howmany__→ NUM    ~90% precision (how + quantifier)
#   __feat_define__ → DESC  100% precision

_YESNO_STARTERS = frozenset({
    "is", "does", "do", "was", "are", "were", "has", "will", "did", "should",
})
_WHO_WORDS   = frozenset({"who", "whom", "whose"})
_HOW_MANY    = frozenset({"many", "much", "often", "long", "far", "old", "tall",
                           "deep", "wide", "fast", "big", "large", "heavy", "high"})
_DESC_VERBS  = frozenset({"define", "explain", "meaning"})
_ABBR_WORDS  = frozenset({"abbreviation", "acronym", "initials"})


import re as _re

_PUNCT_RE = _re.compile(r"[^\w\s]", flags=_re.UNICODE)


def normalize_punctuated(text: str) -> str:
    """Lowercase — keeps punctuation (written/typed text)."""
    return text.lower().strip()


def normalize_unpunctuated(text: str) -> str:
    """Lowercase + strip all punctuation (ASR/spoken text)."""
    return _PUNCT_RE.sub("", text.lower()).strip()


class TextNormalizer(BaseEstimator, TransformerMixin):
    """Normalize text before TF-IDF. punctuated=True keeps punct, False strips it."""

    def __init__(self, punctuated: bool = True) -> None:
        self.punctuated = punctuated

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        fn = normalize_punctuated if self.punctuated else normalize_unpunctuated
        return [fn(t) for t in X]


def inject_eat_features(text: str) -> str:
    """Prepend high-precision categorical signal tokens to *text*.

    Expects already-normalised (lowercased) input. Called on both training
    texts and at inference time (before ONNX session). The ONNX model's
    TF-IDF vocabulary includes these __feat_*__ unigrams, so they contribute
    just like any other feature — no custom ONNX op needed.
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
        if second in _HOW_MANY:
            tokens.append("__feat_howmany__")
        else:
            tokens.append("__feat_howother__")
    if any(w in text for w in _DESC_VERBS):
        tokens.append("__feat_define__")
    if any(w in text for w in _ABBR_WORDS):
        tokens.append("__feat_abbr__")

    return (" ".join(tokens) + " " + text) if tokens else text


class EATTextPreprocessor(BaseEstimator, TransformerMixin):
    """Normalize + inject categorical tokens.

    punctuated=True (default): lowercase, keep punctuation — for typed/written text.
    punctuated=False: lowercase, strip punctuation — for ASR/spoken text.
    """

    def __init__(self, punctuated: bool = True) -> None:
        self.punctuated = punctuated

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        fn = normalize_punctuated if self.punctuated else normalize_unpunctuated
        return [inject_eat_features(fn(text)) for text in X]


class LemmatizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, stemmer=None):
        self.stemmer = stemmer

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        return normalize_text(X, stemmer=self.stemmer, **transform_params)


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

    def save(self, path: str) -> None:
        import joblib

        if self.clf is None:
            raise RuntimeError("Classifier not trained. Call train() first.")
        joblib.dump(self.clf, path)

    def save_onnx(self, path: str) -> None:
        """Convert and save as ONNX model."""
        if self.clf is None:
            raise RuntimeError("Classifier not trained. Call train() first.")

        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import StringTensorType
            import onnx
        except ImportError as e:
            raise ImportError(
                "skl2onnx and onnx required. Install: pip install skl2onnx onnx"
            ) from e

        from sklearn.svm import LinearSVC

        import json

        initial_type = [("input", StringTensorType([None]))]
        onnx_model = convert_sklearn(
            self.clf, initial_types=initial_type, options={LinearSVC: {"nocl": True}}
        )
        if isinstance(onnx_model, tuple):
            onnx_model = onnx_model[0]

        # Embed class labels as metadata so inference needs no sidecar.
        classes = list(getattr(self.clf, "classes_", []))
        if not classes:
            # Pipeline: access the final estimator's classes.
            try:
                classes = list(list(self.clf.steps)[-1][1].classes_)
            except Exception:
                pass
        if classes:
            meta = onnx_model.metadata_props.add()
            meta.key = "classes"
            meta.value = json.dumps(classes)

        onnx.save_model(onnx_model, path)


class LinearSVCClassifier(TrainableClassifier):
    """Linear SVM text classifier with optional categorical features.

    Args:
        pipeline_id: Identifier for the pipeline (default "linear-svc")
        categorical_transformer: Optional sklearn transformer for categorical features.
            If provided, uses FeatureUnion([TF-IDF, categorical]) instead of TF-IDF alone.
    """

    def __init__(self, pipeline_id: str = "linear-svc", categorical_transformer=None) -> None:
        super().__init__(pipeline_id)
        self.categorical_transformer = categorical_transformer

    @property
    def pipeline(self) -> list:
        # Cap TF-IDF vocabulary to keep total features <= 200 when combined with categorical
        # With 14 categorical features, target ~180 TF-IDF features = 194 total
        tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,  # Back to 1 for better signal
            max_df=0.4,  # Back to 0.4
            max_features=180,  # Cap vocabulary size
        )

        if self.categorical_transformer is None:
            # Standard: TF-IDF only
            features = tfidf
        else:
            # FeatureUnion: TF-IDF + categorical features
            features = FeatureUnion([
                ("tfidf", tfidf),
                ("categorical", self.categorical_transformer),
            ])

        return [
            ("features", features),
            ("clf", _LinearSVC()),
        ]


class CalibratedLinearSVCClassifier(TrainableClassifier):
    """Platt-calibrated Linear SVM classifier with optional categorical features.

    Args:
        pipeline_id: Identifier string (default "calibrated-svc").
        use_categorical: When True (default), fuses EATFeatureTransformer binary
            indicators with TF-IDF via FeatureUnion for a free accuracy boost.

    Wraps LinearSVC in CalibratedClassifierCV(method='sigmoid', cv=3) so that
    ONNX output[1] is a genuine probability vector (values in [0,1], sum to 1)
    rather than raw decision-function distances.

    Export: use save_onnx() — requires options={CalibratedClassifierCV: {"zipmap": False}}
    to get a plain float32 matrix instead of a ZipMap dict.
    """

    def __init__(self, pipeline_id: str = "calibrated-svc",
                 punctuated: bool = True) -> None:
        """
        Args:
            punctuated: True (default) for written/typed text (keeps punctuation);
                        False for ASR/spoken text (strips all punctuation).
        """
        super().__init__(pipeline_id)
        self.punctuated = punctuated

    @property
    def pipeline(self) -> list:
        tfidf = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True,
            max_features=20_000,
            # \w matches word chars including underscores → picks up __feat_*__ tokens
            token_pattern=r"(?u)\b\w+\b",
        )
        clf = CalibratedClassifierCV(
            _LinearSVC(C=1.0, max_iter=2000), cv=3, method="sigmoid"
        )
        prep = EATTextPreprocessor(punctuated=self.punctuated)
        return [("prep", prep), ("tfidf", tfidf), ("clf", clf)]

    def save_onnx(self, path: str) -> None:
        """Export to ONNX with zipmap=False so output[1] is a float32 matrix."""
        if self.clf is None:
            raise RuntimeError("Classifier not trained. Call train() first.")

        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import StringTensorType
            import onnx, json
        except ImportError as e:
            raise ImportError(
                "skl2onnx and onnx required. Install: pip install skl2onnx onnx"
            ) from e

        initial_type = [("input", StringTensorType([None]))]
        options = {CalibratedClassifierCV: {"zipmap": False}}
        onnx_model = convert_sklearn(self.clf, initial_types=initial_type, options=options)
        if isinstance(onnx_model, tuple):
            onnx_model = onnx_model[0]

        # Embed sorted class labels as metadata for inference-time lookup.
        try:
            classes = list(self.clf.steps[-1][1].classes_)
        except Exception:
            classes = []
        if classes:
            meta = onnx_model.metadata_props.add()
            meta.key = "classes"
            meta.value = json.dumps(classes)
        # Mark as calibrated so inference code knows output[1] is probabilities.
        flag = onnx_model.metadata_props.add()
        flag.key = "calibrated"
        flag.value = "true"

        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        onnx.save_model(onnx_model, path)


class LogRegClassifier(TrainableClassifier):
    """Logistic Regression text classifier."""

    @property
    def pipeline(self) -> list:
        return [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.4)),
            ("clf", _LogisticRegression(solver="lbfgs", max_iter=1000)),
        ]


class RandomForestClassifier(TrainableClassifier):
    """Random Forest text classifier."""

    @property
    def pipeline(self) -> list:
        return [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.4)),
            ("clf", _RandomForestClassifier()),
        ]


class NaiveBayesClassifier(TrainableClassifier):
    """Naive Bayes text classifier."""

    @property
    def pipeline(self) -> list:
        return [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.4)),
            ("clf", _MultinomialNB()),
        ]


class PassiveAggressiveClassifier(TrainableClassifier):
    """Passive Aggressive text classifier."""

    @property
    def pipeline(self) -> list:
        return [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.4)),
            ("clf", _PassiveAggressiveClassifier(loss="hinge")),
        ]


class SGDClassifier(TrainableClassifier):
    """Stochastic Gradient Descent text classifier."""

    @property
    def pipeline(self) -> list:
        return [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.4)),
            ("clf", _SGDClassifier(loss="hinge", penalty="l2", alpha=1e-3)),
        ]


class PerceptronClassifier(TrainableClassifier):
    """Perceptron text classifier."""

    @property
    def pipeline(self) -> list:
        return [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.4)),
            ("clf", _Perceptron()),
        ]


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

    name: str  # set after construction

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
        """Return embedding matrix for *texts*."""
        return self._m2v_model.encode(texts)

    def _features(self, texts: List[str], fit: bool = False):
        """Return feature matrix (embeddings, optionally fused with TF-IDF)."""
        import numpy as np
        from scipy.sparse import hstack, issparse

        emb = self._encode(texts)
        if not self.use_tfidf_fusion:
            return emb

        if fit:
            self._tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.4)
            tfidf_feat = self._tfidf.fit_transform(texts)
        else:
            tfidf_feat = self._tfidf.transform(texts)

        # hstack handles sparse + dense by converting dense to sparse
        return hstack([tfidf_feat, emb])

    def train(self, train_data: List[str], target_data: List[str]) -> None:
        """Fit the classifier on *train_data* / *target_data*."""
        from model2vec import StaticModel

        self._m2v_model = StaticModel.from_pretrained(self.model_name)
        X = self._features(train_data, fit=True)
        self._clf = _LinearSVC()
        self._clf.fit(X, target_data)
        self._classes = list(self._clf.classes_)

    def predict(self, texts: List[str]) -> List[str]:
        """Return predicted labels for *texts*."""
        if self._clf is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return list(self._clf.predict(self._features(texts)))

    def save(self, directory: str) -> None:
        """Save model artefacts to *directory*.

        Writes:
        - ``clf.joblib`` — the fitted LinearSVC
        - ``tfidf.joblib`` — the TF-IDF vectorizer (only when fusion is active)
        - ``model2vec/`` — the model2vec StaticModel directory
        - ``meta.json`` — metadata (model name, fusion flag, classes)
        """
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
        """Load a previously saved Model2VecClassifier from *directory*."""
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
