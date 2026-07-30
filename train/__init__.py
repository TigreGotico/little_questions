"""Training module for little_questions classifiers.

Install with: pip install little-questions[train]

Example:
    from train.classifiers import CalibratedLinearSVCClassifier

    clf = CalibratedLinearSVCClassifier()
    clf.train(train_data, labels)
    clf.save_onnx("eat53_svm_cal_EN_0.9.0.onnx")
"""

from train.classifiers import (
    LinearSVCClassifier,
    CalibratedLinearSVCClassifier,
    LogRegClassifier,
    SGDClassifier,
    Model2VecClassifier,
)

__all__ = [
    "LinearSVCClassifier",
    "CalibratedLinearSVCClassifier",
    "LogRegClassifier",
    "SGDClassifier",
    "Model2VecClassifier",
]
