#!/usr/bin/env python3
"""Benchmark sentence-type ONNX classifiers across all supported languages.

Loads the bundled ONNX models and evaluates them against the labelled
sentence-type datasets in ``train/clean_data/``.  Outputs text reports and
six publication-quality plots saved to ``train/reports/sentence_type/``.

Usage::

    python -m train.benchmark_sentence_type              # all 7 languages
    python -m train.benchmark_sentence_type --lang en    # single language
    python -m train.benchmark_sentence_type --no-plot    # skip plots
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from os.path import dirname, join
from pathlib import Path

import numpy as np

LOG = logging.getLogger(__name__)

DATA_DIR = join(dirname(__file__), "clean_data")
REPORTS_DIR = join(dirname(__file__), "reports", "sentence_type")
BUNDLED_DIR = join(dirname(dirname(__file__)), "little_questions", "models", "sentence_type")
VERSION = "0.8.0"

LANGUAGES = ["en", "de", "es", "fr", "it", "nl", "pt"]
LABELS = ["command", "exclamation", "polar_question", "request", "statement", "wh_question"]


# ---------------------------------------------------------------------------
# ONNX inference wrapper
# ---------------------------------------------------------------------------

class SentenceTypeOnnxScorer:
    def __init__(self, onnx_path: str) -> None:
        import onnxruntime as rt
        self.name = Path(onnx_path).stem
        self._sess = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self._input_name = self._sess.get_inputs()[0].name
        meta = self._sess.get_modelmeta().custom_metadata_map
        self.classes: list[str] = json.loads(meta.get("classes", "[]"))

    def predict(self, text: str) -> str:
        result = self._sess.run(None, {self._input_name: [text]})
        idx = int(result[0][0])
        return self.classes[idx]

    def predict_proba(self, text: str) -> dict[str, float]:
        result = self._sess.run(None, {self._input_name: [text]})
        proba = result[1][0]
        return {c: float(proba[i]) for i, c in enumerate(self.classes)}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_data(lang: str) -> tuple[list[str], list[str]]:
    path = join(DATA_DIR, f"sentence_types_{lang.upper()}.txt")
    x, y = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        label, _, text = line.partition(":")
        label = label.strip()
        text = text.strip()
        if label and text:
            x.append(text)
            y.append(label)
    return x, y


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(scorer: SentenceTypeOnnxScorer, x: list[str], y: list[str]) -> dict:
    from sklearn.metrics import (
        accuracy_score, f1_score, classification_report, confusion_matrix,
    )

    y_pred = [scorer.predict(t) for t in x]
    acc = accuracy_score(y, y_pred)
    mf1 = f1_score(y, y_pred, average="macro", zero_division=0)
    wf1 = f1_score(y, y_pred, average="weighted", zero_division=0)
    report = classification_report(y, y_pred, zero_division=0)
    labels = sorted(set(y) | set(y_pred))
    cm = confusion_matrix(y, y_pred, labels=labels)
    per_class = {
        lbl: float(f1_score([yi == lbl for yi in y], [yp == lbl for yp in y_pred],
                             average="binary", zero_division=0))
        for lbl in labels
    }
    return {
        "accuracy": acc, "macro_f1": mf1, "weighted_f1": wf1,
        "report": report, "confusion_matrix": cm.tolist(),
        "labels": labels, "per_class_f1": per_class,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_overview(results: dict[str, dict], out_dir: str) -> str:
    import matplotlib.pyplot as plt

    langs = list(results)
    accs = [results[l]["accuracy"] for l in langs]
    mf1s = [results[l]["macro_f1"] for l in langs]
    x = np.arange(len(langs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, accs, width, label="Accuracy", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, mf1s, width, label="Macro F1", color="#DD8452")
    ax.set_xlabel("Language")
    ax.set_ylabel("Score")
    ax.set_title("Sentence-Type Classifier — All Languages")
    ax.set_xticks(x)
    ax.set_xticklabels([l.upper() for l in langs])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.yaxis.grid(True, alpha=0.3)

    for bar in bars1:
        ax.annotate(f"{bar.get_height():.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    for bar in bars2:
        ax.annotate(f"{bar.get_height():.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    fig.tight_layout()
    path = join(out_dir, "benchmark_sentence_type_overview.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion(result: dict, lang: str, out_dir: str) -> str:
    import matplotlib.pyplot as plt

    cm = np.array(result["confusion_matrix"])
    labels = result["labels"]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Sentence-Type Confusion — {lang.upper()}")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if cm_norm[i, j] > 0.6 else "black")
    fig.tight_layout()
    path = join(out_dir, f"benchmark_sentence_type_{lang}_confusion.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_per_class_f1(result: dict, lang: str, out_dir: str) -> str:
    import matplotlib.pyplot as plt

    per_class = result["per_class_f1"]
    labels_sorted = sorted(per_class, key=per_class.__getitem__)
    scores = [per_class[l] for l in labels_sorted]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#d62728" if s < 0.7 else "#2ca02c" if s >= 0.9 else "#ff7f0e" for s in scores]
    ax.barh(labels_sorted, scores, color=colors)
    ax.set_xlabel("F1 score")
    ax.set_title(f"Sentence-Type Per-Class F1 — {lang.upper()}")
    ax.set_xlim(0, 1.05)
    ax.axvline(0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    for i, (score, label) in enumerate(zip(scores, labels_sorted)):
        ax.text(score + 0.01, i, f"{score:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    path = join(out_dir, f"benchmark_sentence_type_{lang}_per_class_f1.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_macro_f1_heatmap(results: dict[str, dict], out_dir: str) -> str:
    """Per-class F1 heatmap across all languages."""
    import matplotlib.pyplot as plt

    langs = list(results)
    all_labels = sorted({lbl for r in results.values() for lbl in r["per_class_f1"]})
    data = np.array([[results[l]["per_class_f1"].get(lbl, 0.0) for lbl in all_labels]
                     for l in langs])

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    ax.set_xticks(range(len(all_labels)))
    ax.set_yticks(range(len(langs)))
    ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels([l.upper() for l in langs], fontsize=9)
    ax.set_title("Per-Class F1 by Language")
    for i in range(len(langs)):
        for j in range(len(all_labels)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    path = join(out_dir, "benchmark_sentence_type_per_class_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def benchmark(langs: list[str], plot: bool = True) -> None:
    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    print(f"\n{'='*70}")
    print("Sentence-Type Benchmark")
    print(f"{'='*70}\n")

    for lang in langs:
        model_path = join(BUNDLED_DIR, f"sentence_type_{lang.upper()}_{VERSION}.onnx")
        if not Path(model_path).exists():
            # fall back to user cache
            model_path = os.path.expanduser(
                f"~/.local/share/little_questions/sentence_type/sentence_type_{lang.upper()}_{VERSION}.onnx"
            )
        if not Path(model_path).exists():
            print(f"[{lang.upper()}] No model found — skipping")
            continue

        x, y = load_data(lang)
        scorer = SentenceTypeOnnxScorer(model_path)
        result = evaluate_model(scorer, x, y)
        results[lang] = result

        print(f"[{lang.upper()}]  accuracy={result['accuracy']:.4f}  "
              f"macro_f1={result['macro_f1']:.4f}  weighted_f1={result['weighted_f1']:.4f}")

        report_path = join(REPORTS_DIR, f"sentence_type_{lang.upper()}_{VERSION}.txt")
        Path(report_path).write_text(result["report"], encoding="utf-8")

        json_path = join(REPORTS_DIR, f"sentence_type_{lang.upper()}_{VERSION}_benchmark.json")
        payload = {k: v for k, v in result.items() if k != "report"}
        Path(json_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not results:
        print("No results — no models or data found.")
        return

    print(f"\n{'='*70}")
    print(f"  {'Lang':<6} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*12}")
    for lang, r in results.items():
        print(f"  {lang.upper():<6} {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f} {r['weighted_f1']:>12.4f}")

    if plot:
        print(f"\nSaving plots to {REPORTS_DIR}/")
        p = plot_overview(results, REPORTS_DIR)
        print(f"  {p}")
        p = plot_macro_f1_heatmap(results, REPORTS_DIR)
        print(f"  {p}")
        for lang in results:
            p = plot_confusion(results[lang], lang, REPORTS_DIR)
            print(f"  {p}")
            p = plot_per_class_f1(results[lang], lang, REPORTS_DIR)
            print(f"  {p}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark sentence-type ONNX classifiers")
    parser.add_argument("--lang", default=None, help="Single language code (default: all)")
    parser.add_argument("--no-plot", dest="plot", action="store_false", default=True)
    args = parser.parse_args()

    langs = [args.lang] if args.lang else LANGUAGES
    benchmark(langs, plot=args.plot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
