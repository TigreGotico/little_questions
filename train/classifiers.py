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
from sklearn.pipeline import Pipeline
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

        initial_type = [("input", StringTensorType([None]))]
        onnx_model = convert_sklearn(
            self.clf, initial_types=initial_type, options={LinearSVC: {"nocl": True}}
        )
        if isinstance(onnx_model, tuple):
            onnx.save_model(onnx_model[0], path)
        else:
            onnx.save_model(onnx_model, path)


class LinearSVCClassifier(TrainableClassifier):
    """Linear SVM text classifier."""

    @property
    def pipeline(self) -> list:
        return [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.4)),
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
