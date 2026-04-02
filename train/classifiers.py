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
