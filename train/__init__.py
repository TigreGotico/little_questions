"""Training module for little_questions classifiers.

Install with: pip install little-questions[train]

Example:
    from train.classifiers import LinearSVCClassifier

    clf = LinearSVCClassifier("en")
    clf.train(train_data, labels)
    clf.save_onnx("questions52_svm_EN_0.8.0.onnx")
"""

from train.classifiers import LinearSVCClassifier
from train.classifiers import LogRegClassifier
from train.classifiers import RandomForestClassifier
from train.classifiers import NaiveBayesClassifier
from train.classifiers import PassiveAggressiveClassifier
from train.classifiers import SGDClassifier
from train.classifiers import PerceptronClassifier

__all__ = [
    "LinearSVCClassifier",
    "LogRegClassifier",
    "RandomForestClassifier",
    "NaiveBayesClassifier",
    "PassiveAggressiveClassifier",
    "SGDClassifier",
    "PerceptronClassifier",
]
