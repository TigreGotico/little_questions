#!/usr/bin/env python3
"""Benchmark EAT ONNX models — inference via onnxruntime only, no sklearn required.

Usage::

    python -m train.benchmark_eat                       # benchmark all trained models
    python -m train.benchmark_eat --classes 53          # 53-class only
    python -m train.benchmark_eat --model svm           # single model type
    python -m train.benchmark_eat --dataset path/to/EAT.tsv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from os.path import join
from pathlib import Path

import numpy as np

LOG = logging.getLogger(__name__)

MODEL_DIR = os.path.expanduser("~/.local/share/little_questions/eat")
REPORTS_DIR = join(os.path.dirname(__file__), "reports", "eat")
VERSION = "0.9.0"

HF_DATASET = "TigreGotico/EAT"


# ---------------------------------------------------------------------------
# ONNX inference wrapper — satisfies the Scorer protocol in metrics.py
# ---------------------------------------------------------------------------

class OnnxScorer:
    """Wraps an EAT ONNX model for use with train.metrics.evaluate()."""

    def __init__(self, onnx_path: str) -> None:
        import onnxruntime as rt

        self.name = Path(onnx_path).stem
        self._sess = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self._input_name = self._sess.get_inputs()[0].name
        self._output_name = self._sess.get_outputs()[0].name
        self._output_type = self._sess.get_outputs()[0].type

        meta = self._sess.get_modelmeta().custom_metadata_map
        raw = meta.get("classes", "[]")
        self._classes: list[str] = json.loads(raw)

    def _decode(self, raw) -> str:
        """Convert ONNX output (int index or bytes/str label) to a string label."""
        if isinstance(raw, (int, np.integer)):
            return self._classes[int(raw)] if self._classes else str(raw)
        if isinstance(raw, bytes):
            return raw.decode("utf-8")
        return str(raw)

    def predict(self, text: str) -> str:
        inp = np.array([text], dtype=object)
        result = self._sess.run([self._output_name], {self._input_name: inp})
        return self._decode(result[0][0])

    def predict_batch(self, texts: list[str]) -> list[str]:
        inp = np.array(texts, dtype=object)
        result = self._sess.run([self._output_name], {self._input_name: inp})
        return [self._decode(r) for r in result[0]]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_split(dataset_path: str | None, classes: int) -> tuple[list[str], list[str]]:
    """Return the same 15 % stratified test split used during training."""
    from sklearn.model_selection import train_test_split

    if dataset_path:
        from train.load_eat import load_eat
        x, y = load_eat(dataset_path, classes=classes)
    else:
        from train.load_eat import load_eat_hf
        x, y = load_eat_hf(classes=classes)

    if classes == 7:
        y = [lbl.split(":")[0] for lbl in y]

    _, x_test, _, y_test = train_test_split(
        x, y, test_size=0.15, stratify=y, random_state=42
    )
    return x_test, y_test


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_overview(all_results: list, save_dir: str) -> None:
    """Grouped bar chart: all models × accuracy/macro_f1, one panel per class count."""
    import matplotlib.pyplot as plt

    for n_cls in (53, 7):
        subset = [r for r in all_results if r["classes"] == n_cls]
        if not subset:
            continue
        names = [r["scorer_name"] for r in subset]
        accs = [r["accuracy"] for r in subset]
        f1s = [r["macro_f1"] for r in subset]

        x = np.arange(len(names))
        width = 0.35
        fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.6), 4))
        bars_a = ax.bar(x - width / 2, accs, width, label="Accuracy", color="steelblue")
        bars_f = ax.bar(x + width / 2, f1s, width, label="Macro F1", color="darkorange")
        for bars in (bars_a, bars_f):
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                        f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title(f"EAT {n_cls}-class — accuracy & macro F1 per model")
        ax.legend()
        fig.tight_layout()
        path = join(save_dir, f"benchmark_eat{n_cls}_overview.png")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
        plt.close(fig)


def plot_comparison(all_results: list, save_dir: str) -> None:
    """Line plot: macro F1 vs model type, one line per class granularity."""
    import matplotlib.pyplot as plt

    model_types = list(dict.fromkeys(r["model_type"] for r in all_results))
    f1_53 = {r["model_type"]: r["macro_f1"] for r in all_results if r["classes"] == 53}
    f1_7  = {r["model_type"]: r["macro_f1"] for r in all_results if r["classes"] == 7}

    fig, ax = plt.subplots(figsize=(max(6, len(model_types) * 1.4), 4))
    if f1_53:
        ax.plot(model_types, [f1_53.get(m, 0) for m in model_types],
                marker="o", linewidth=2, label="53-class", color="steelblue")
        for mt in model_types:
            ax.annotate(f"{f1_53.get(mt, 0):.3f}", (mt, f1_53.get(mt, 0)),
                        textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    if f1_7:
        ax.plot(model_types, [f1_7.get(m, 0) for m in model_types],
                marker="s", linewidth=2, label="7-class", color="darkorange")
        for mt in model_types:
            ax.annotate(f"{f1_7.get(mt, 0):.3f}", (mt, f1_7.get(mt, 0)),
                        textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro F1")
    ax.set_title("EAT — macro F1 by model type & label granularity")
    ax.legend()
    fig.tight_layout()
    path = join(save_dir, "benchmark_eat_model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_confusion(eval_result, n_cls: int, save_dir: str) -> None:
    path = join(save_dir, f"benchmark_eat{n_cls}_confusion.png")
    # Use a smaller font for 53-class to keep it readable
    fontsize = 6 if n_cls > 10 else 9
    fig_size = max(10, n_cls * 0.55)

    import matplotlib.pyplot as plt
    cm = np.array(eval_result.conf_matrix, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(eval_result.labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(eval_result.labels, rotation=60, ha="right", fontsize=fontsize)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(eval_result.labels, fontsize=fontsize)

    thresh = 0.5
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val = cm_norm[i, j]
            if val > 0.01:  # skip near-zero cells to reduce clutter
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white" if val > thresh else "black", fontsize=fontsize)

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(f"{eval_result.scorer_name} — {n_cls}-class confusion matrix (normalised)")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_per_class_f1(eval_result, n_cls: int, save_dir: str) -> None:
    path = join(save_dir, f"benchmark_eat{n_cls}_per_class_f1.png")
    labels = [l for l in eval_result.labels if l in eval_result.per_class]
    f1s = [eval_result.per_class[l].get("f1-score", 0.0) for l in labels]
    # Sort ascending so weakest classes are most visible at left
    pairs = sorted(zip(f1s, labels))
    f1s_sorted = [p[0] for p in pairs]
    labels_sorted = [p[1] for p in pairs]

    import matplotlib.pyplot as plt
    fig_w = max(8, len(labels) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    colors = ["#d62728" if v < 0.7 else "#2ca02c" if v >= 0.9 else "steelblue"
              for v in f1s_sorted]
    bars = ax.bar(labels_sorted, f1s_sorted, color=colors, edgecolor="white")
    ax.axhline(eval_result.macro_f1, color="red", linestyle="--", linewidth=1,
               label=f"macro avg = {eval_result.macro_f1:.3f}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-score")
    ax.set_title(f"{eval_result.scorer_name} — per-class F1 ({n_cls} classes, sorted ascending)")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=60, labelsize=7 if n_cls > 15 else 9)
    for bar, val in zip(bars, f1s_sorted):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=6 if n_cls > 20 else 8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MODEL_TYPES = ["svm", "logreg", "sgd"]


def _model_path(model_type: str, n_cls: int) -> str:
    name = f"eat{n_cls}_{model_type}_EN_{VERSION}.onnx"
    return join(MODEL_DIR, name)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Benchmark EAT ONNX models")
    parser.add_argument("--dataset", default=None,
                        help="Path to local EAT.tsv (default: load from HuggingFace)")
    parser.add_argument("--classes", type=int, choices=[7, 53],
                        help="Only benchmark this class granularity")
    parser.add_argument("--model", choices=MODEL_TYPES,
                        help="Only benchmark this model type")
    args = parser.parse_args()

    from train.metrics import evaluate, compare

    class_variants = [args.classes] if args.classes else [53, 7]
    model_types = [args.model] if args.model else MODEL_TYPES

    all_summary: list[dict] = []
    best_per_cls: dict[int, tuple] = {}  # n_cls → (best_macro_f1, eval_result)

    for n_cls in class_variants:
        print(f"\nLoading test split ({n_cls}-class)…")
        x_test, y_test = load_test_split(args.dataset, n_cls)
        print(f"  {len(x_test)} test samples, {len(set(y_test))} classes")

        cls_results = []
        for mt in model_types:
            path = _model_path(mt, n_cls)
            if not os.path.exists(path):
                LOG.warning("Model not found, skipping: %s", path)
                continue

            scorer = OnnxScorer(path)
            # Batch predict for speed, then wrap in a result via evaluate()
            y_pred_batch = scorer.predict_batch(x_test)
            # Monkey-patch predict so evaluate() works without re-running batch
            _pred_iter = iter(y_pred_batch)
            scorer.predict = lambda _t, _it=_pred_iter: next(_it)
            result = evaluate(scorer, x_test, y_test,
                              dataset=HF_DATASET, lang="en")
            cls_results.append(result)
            all_summary.append({
                "scorer_name": scorer.name,
                "model_type": mt,
                "classes": n_cls,
                "accuracy": result.accuracy,
                "macro_f1": result.macro_f1,
                "weighted_f1": result.weighted_f1,
            })
            # Save JSON report
            result.save(join(REPORTS_DIR, f"{scorer.name}_benchmark.json"))

            if n_cls not in best_per_cls or result.macro_f1 > best_per_cls[n_cls][0]:
                best_per_cls[n_cls] = (result.macro_f1, result)

        if cls_results:
            print(f"\n--- {n_cls}-class comparison ---")
            compare(cls_results)

    # Plots
    if all_summary:
        Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
        plot_overview(all_summary, REPORTS_DIR)
        plot_comparison(all_summary, REPORTS_DIR)

    for n_cls, (_, best) in best_per_cls.items():
        print(f"\nBest {n_cls}-class model: {best.scorer_name}  macro_f1={best.macro_f1:.4f}")
        plot_confusion(best, n_cls, REPORTS_DIR)
        plot_per_class_f1(best, n_cls, REPORTS_DIR)

    print("\nDone. Reports and plots saved to:", REPORTS_DIR)


if __name__ == "__main__":
    main()
