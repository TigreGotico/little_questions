"""Evaluation metrics for little_questions classifiers.

Provides a unified evaluation harness that works for any scorer/classifier
with a predict(text) -> str interface — trained models and baselines alike.

Usage::

    from train.metrics import evaluate, EvalResult
    from train.baselines import PunctuationScorer

    result = evaluate(PunctuationScorer(), texts, labels)
    print(result)
    result.plot_confusion_matrix()
    result.save("reports/punctuation_baseline.json")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

LOG = logging.getLogger(__name__)


class Scorer(Protocol):
    """Minimal interface required for evaluation."""

    name: str

    def predict(self, text: str) -> str:
        ...


@dataclass
class EvalResult:
    """Aggregated evaluation result for a single scorer/dataset combination.

    Attributes:
        scorer_name: Name of the scorer.
        dataset: Dataset file path or description.
        lang: Language code.
        labels: Class labels (sorted).
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        accuracy: Overall accuracy.
        macro_f1: Macro-averaged F1 score.
        weighted_f1: Weighted F1 score.
        per_class: Per-class precision/recall/F1/support.
        conf_matrix: Confusion matrix (rows=true, cols=pred).
    """

    scorer_name: str
    dataset: str
    lang: str
    labels: List[str]
    y_true: List[str]
    y_pred: List[str]
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: Dict[str, Dict[str, float]]
    conf_matrix: List[List[int]]

    def __str__(self) -> str:
        lines = [
            f"Scorer : {self.scorer_name}",
            f"Dataset: {self.dataset}  (lang={self.lang})",
            f"Samples: {len(self.y_true)}",
            f"",
            classification_report(
                self.y_true, self.y_pred, labels=self.labels, zero_division=0
            ),
            f"Accuracy : {self.accuracy:.4f}",
            f"Macro F1 : {self.macro_f1:.4f}",
            f"Wtd   F1 : {self.weighted_f1:.4f}",
        ]
        return "\n".join(lines)

    def save(self, path: str) -> None:
        """Serialise result to a JSON file."""
        data = {
            "scorer": self.scorer_name,
            "dataset": self.dataset,
            "lang": self.lang,
            "labels": self.labels,
            "n_samples": len(self.y_true),
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "per_class": self.per_class,
            "confusion_matrix": self.conf_matrix,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        LOG.info("Saved eval result: %s", path)

    def plot_confusion_matrix(
        self,
        title: Optional[str] = None,
        normalise: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot the confusion matrix using matplotlib.

        Args:
            title: Plot title (defaults to scorer name + dataset).
            normalise: Normalise rows to [0, 1] (shows recall per class).
            save_path: If given, save figure here instead of showing it.
        """
        import matplotlib.pyplot as plt

        cm = np.array(self.conf_matrix, dtype=float)
        if normalise:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
            fmt = ".2f"
            vmax = 1.0
        else:
            fmt = "d"
            cm = cm.astype(int)
            vmax = None

        fig, ax = plt.subplots(figsize=(max(6, len(self.labels)), max(5, len(self.labels))))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax)

        tick_marks = np.arange(len(self.labels))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(self.labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(self.labels, fontsize=9)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = f"{cm[i, j]:{fmt}}" if fmt != "d" else str(int(np.array(self.conf_matrix)[i, j]))
                ax.text(j, i, val, ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=8)

        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        ax.set_title(title or f"{self.scorer_name} — {Path(self.dataset).name}")
        fig.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            LOG.info("Saved confusion matrix: %s", save_path)
            plt.close(fig)
        else:
            plt.show()

    def plot_per_class_bars(
        self,
        metric: str = "f1-score",
        save_path: Optional[str] = None,
    ) -> None:
        """Bar chart of per-class precision, recall, and F1.

        Args:
            metric: One of ``"precision"``, ``"recall"``, ``"f1-score"``.
            save_path: If given, save figure here instead of showing it.
        """
        import matplotlib.pyplot as plt

        labels = [l for l in self.labels if l in self.per_class]
        values = [self.per_class[l].get(metric, 0.0) for l in labels]

        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), 4))
        bars = ax.bar(labels, values, color="steelblue", edgecolor="white")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"{self.scorer_name} — per-class {metric}")
        ax.axhline(self.macro_f1, color="red", linestyle="--", linewidth=1,
                   label=f"macro avg = {self.macro_f1:.2f}")
        ax.legend(fontsize=8)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

        fig.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            LOG.info("Saved per-class bar chart: %s", save_path)
            plt.close(fig)
        else:
            plt.show()


def evaluate(
    scorer: Any,
    texts: List[str],
    labels: List[str],
    *,
    scorer_name: Optional[str] = None,
    dataset: str = "unknown",
    lang: str = "en",
) -> EvalResult:
    """Evaluate *scorer* on *(texts, labels)* and return an :class:`EvalResult`.

    Args:
        scorer: Any object with ``predict(text: str) -> str``.
        texts: Input utterances.
        labels: Ground-truth sentence-type labels.
        scorer_name: Override the name used in reports (defaults to
            ``scorer.name`` or the class name).
        dataset: Label for the dataset (used in report title).
        lang: Language code.

    Returns:
        :class:`EvalResult` with accuracy, F1, per-class stats, and confusion matrix.
    """
    name = scorer_name or getattr(scorer, "name", type(scorer).__name__)
    LOG.info("Evaluating %s on %d samples (%s)...", name, len(texts), dataset)

    y_pred = [scorer.predict(t) for t in texts]
    class_labels = sorted(set(labels) | set(y_pred))

    report_dict = classification_report(
        labels, y_pred, labels=class_labels, output_dict=True, zero_division=0
    )
    per_class = {
        k: v for k, v in report_dict.items()
        if k not in ("accuracy", "macro avg", "weighted avg")
        and isinstance(v, dict)
    }

    return EvalResult(
        scorer_name=name,
        dataset=dataset,
        lang=lang,
        labels=class_labels,
        y_true=list(labels),
        y_pred=y_pred,
        accuracy=accuracy_score(labels, y_pred),
        macro_f1=f1_score(labels, y_pred, average="macro", zero_division=0),
        weighted_f1=f1_score(labels, y_pred, average="weighted", zero_division=0),
        per_class=per_class,
        conf_matrix=confusion_matrix(labels, y_pred, labels=class_labels).tolist(),
    )


def compare(
    results: List[EvalResult],
    save_path: Optional[str] = None,
) -> None:
    """Print a side-by-side comparison table and optionally save a bar chart.

    Args:
        results: List of :class:`EvalResult` objects to compare.
        save_path: If given, save comparison bar chart here.
    """
    print(f"\n{'Scorer':<30} {'Accuracy':>10} {'Macro F1':>10} {'Wtd F1':>10}")
    print("-" * 62)
    for r in sorted(results, key=lambda x: x.macro_f1, reverse=True):
        print(f"{r.scorer_name:<30} {r.accuracy:>10.4f} {r.macro_f1:>10.4f} {r.weighted_f1:>10.4f}")

    if save_path:
        import matplotlib.pyplot as plt

        names = [r.scorer_name for r in results]
        accs = [r.accuracy for r in results]
        f1s = [r.macro_f1 for r in results]

        x = np.arange(len(names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.5), 4))
        ax.bar(x - width / 2, accs, width, label="Accuracy", color="steelblue")
        ax.bar(x + width / 2, f1s, width, label="Macro F1", color="darkorange")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Scorer comparison")
        ax.legend()
        fig.tight_layout()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        LOG.info("Saved comparison chart: %s", save_path)
        plt.close(fig)
