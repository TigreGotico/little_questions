from __future__ import annotations

from typing import Dict, List, Optional

import joblib
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC as _LinearSVC
from sklearn.linear_model import LogisticRegression as _LogisticRegression
from sklearn.naive_bayes import MultinomialNB as _MultinomialNB
from sklearn.ensemble import RandomForestClassifier as _RandomForestClassifier
from sklearn.linear_model import \
    PassiveAggressiveClassifier as _PassiveAggressiveClassifier
from sklearn.linear_model import SGDClassifier as _SGDClassifier
from sklearn.linear_model import Perceptron as _Perceptron

from little_questions.classifiers.lang import get_pipeline
from little_questions.models import get_model_path


# this is meant to be subclassed per language, need to create datasets and
# TODO train a proper classifier
class SentenceScorer:
    """Rule-based sentence-type scorer (language-agnostic fallback).

    Subclass and override the ``*_score`` static methods to implement
    language-specific heuristics.
    """

    @staticmethod
    def predict(text: str) -> str:
        """Return the most likely sentence type for *text*."""
        score = SentenceScorer.score(text)
        best = max(score, key=lambda key: score[key])
        return best

    @staticmethod
    def score(text: str) -> Dict[str, float]:
        """Return a dict mapping sentence type to confidence score."""
        return {
            "question": SentenceScorer.question_score(text),
            "statement": SentenceScorer.statement_score(text),
            "exclamation": SentenceScorer.exclamation_score(text),
            "command": SentenceScorer.command_score(text),
            "request": SentenceScorer.request_score(text)
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
    """Base class for COSC question classifiers backed by a joblib model."""

    def __init__(self, pipeline_id: str) -> None:
        """Initialise with *pipeline_id* (language code, e.g. ``"en"``)."""
        self.pipeline_id: str = pipeline_id.lower().split("-")[0]
        self.clf: Optional[Pipeline] = None

    def train(self, train_data: List[str], target_data: List[str]) -> None:
        """Train the classifier (subclasses must implement)."""
        raise NotImplementedError

    @property
    def pipeline(self) -> list:
        """Return the sklearn pipeline steps as a list of (name, estimator) tuples."""
        return []

    def predict(self, text: List[str]) -> List[str]:
        """Classify *text* and return label list."""
        return self.clf.predict(text)

    def save(self, path: str) -> None:
        """Persist the fitted pipeline to *path* via joblib."""
        joblib.dump(self.clf, path)

    def load_from_file(self, path: Optional[str] = None) -> "Classifier":
        """Load a fitted pipeline from *path* (or the default model path)."""
        path = path or get_model_path(self.pipeline_id)
        self.clf = joblib.load(path)
        return self


class LinearSVCTextClassifier(Classifier):
    def train(self, train_data, target_data):
        self.clf = Pipeline(self.pipeline)
        self.clf.fit(train_data, target_data)
        return self.clf

    @property
    def pipeline(self):
        return [
            ('features', get_pipeline(self.pipeline_id)),
            ('clf', _LinearSVC())
        ]


class LogRegTextClassifier(Classifier):

    def train(self, train_data, target_data):
        self.clf = Pipeline(self.pipeline)
        self.clf.fit(train_data, target_data)
        return self.clf

    @property
    def pipeline(self):
        return [
            ('features', get_pipeline(self.pipeline_id)),
            ('clf', _LogisticRegression(multi_class="multinomial",
                                        solver="lbfgs"))
        ]


class RandomForestTextClassifier(Classifier):

    def train(self, train_data, target_data):
        self.clf = Pipeline(self.pipeline)
        self.clf.fit(train_data, target_data)
        return self.clf

    @property
    def pipeline(self):
        return [
            ('features', get_pipeline(self.pipeline_id)),
            ('clf', _RandomForestClassifier())
        ]


class NaiveBayesTextClassifier(Classifier):

    def train(self, train_data, target_data):
        self.clf = Pipeline(self.pipeline)
        self.clf.fit(train_data, target_data)
        return self.clf

    @property
    def pipeline(self):
        return [
            ('features', get_pipeline(self.pipeline_id)),
            ('clf', _MultinomialNB())
        ]


class PassiveAggressiveTextClassifier(Classifier):

    def train(self, train_data, target_data):
        self.clf = Pipeline(self.pipeline)
        self.clf.fit(train_data, target_data)
        return self.clf

    @property
    def pipeline(self):
        return [
            ('features', get_pipeline(self.pipeline_id)),
            ('clf', _PassiveAggressiveClassifier(loss="hinge"))
        ]


class SGDTextClassifier(Classifier):

    def train(self, train_data, target_data):
        self.clf = Pipeline(self.pipeline)
        self.clf.fit(train_data, target_data)
        return self.clf

    @property
    def pipeline(self):
        return [
            ('features', get_pipeline(self.pipeline_id)),
            ('clf', _SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3))
        ]


class PerceptronTextClassifier(Classifier):

    def train(self, train_data, target_data):
        self.clf = Pipeline(self.pipeline)
        self.clf.fit(train_data, target_data)
        return self.clf

    @property
    def pipeline(self):
        return [
            ('features', get_pipeline(self.pipeline_id)),
            ('clf', _Perceptron())
        ]

