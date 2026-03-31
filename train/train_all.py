#!/usr/bin/env python3
"""Train COSC classifiers for all languages and export to ONNX.

Usage::

    python -m train.train_all               # Train all languages, 52-class
    python -m train.train_all --lang es     # One language only
    python -m train.train_all --classes 6   # 6-class models
    python -m train.train_all --no-onnx     # Save as .pkl instead
    python -m train.train_all --plot        # Plot accuracy per language after training
"""

from __future__ import annotations

import argparse
import logging
import os
from os.path import dirname, join
from pathlib import Path

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from xdg import BaseDirectory as XDG

from train.classifiers import LinearSVCClassifier
from train.utils import load_data

LOG = logging.getLogger(__name__)

DATA_DIR = join(dirname(__file__), "clean_data")
REPORTS_DIR = join(dirname(__file__), "reports")
MODEL_DIR = XDG.save_data_path("little_questions")

LANG_CONFIG: dict[str, dict[str, str]] = {
    "en": {"dataset": "raw_questions_EN_balanced_0.8.0.txt", "suffix": "EN"},
    "es": {"dataset": "raw_questions_ES_googtx0.7.0a1.txt",  "suffix": "ES"},
    "ca": {"dataset": "raw_questions_CA_apertiumtx0.7.0a1.txt", "suffix": "CA"},
    "pt": {"dataset": "raw_questions_PT_googtx0.7.0a1.txt",  "suffix": "PT"},
    "fr": {"dataset": "raw_questions_FR_googtx0.7.0a1.txt",  "suffix": "FR"},
    "de": {"dataset": "raw_questions_DE_0.8.0.txt",          "suffix": "DE"},
    "it": {"dataset": "raw_questions_IT_0.8.0.txt",          "suffix": "IT"},
    "nl": {"dataset": "raw_questions_NL_0.8.0.txt",          "suffix": "NL"},
}


def train_language(
    lang: str,
    classes: int = 52,
    export_onnx: bool = True,
) -> dict | None:
    """Train a COSC classifier for *lang*.

    Returns a dict with ``lang``, ``model_name``, ``accuracy``, ``macro_f1``
    on success, or ``None`` when the dataset is missing.
    """
    config = LANG_CONFIG.get(lang)
    if not config:
        LOG.error("Unknown language: %s", lang)
        return None

    suffix = config["suffix"]
    model_name = f"questions{classes}_svm_{suffix}_0.8.0"

    print(f"\n{'=' * 60}")
    print(f"Training {lang.upper()} — {model_name}")
    print("=" * 60)

    data_path = join(DATA_DIR, config["dataset"])
    if not os.path.exists(data_path):
        LOG.warning("Dataset not found: %s", data_path)
        return None

    x, y = load_data(data_path, classes)
    print(f"Loaded {len(x)} samples  |  {len(set(y))} classes")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.15, stratify=y, random_state=42
    )

    clf = LinearSVCClassifier(lang)
    clf.train(x_train, y_train)

    y_pred = clf.predict(x_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    print(f"\n{report}")

    from sklearn.metrics import accuracy_score, f1_score
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = join(REPORTS_DIR, f"{model_name}.txt")
    Path(report_path).write_text(report, encoding="utf-8")

    if export_onnx:
        onnx_path = join(MODEL_DIR, f"{model_name}.onnx")
        print(f"Exporting ONNX → {onnx_path}")
        try:
            clf.save_onnx(onnx_path)
        except Exception as exc:
            LOG.warning("ONNX export failed (%s); saving pkl", exc)
            clf.save(join(MODEL_DIR, f"{model_name}.pkl"))
    else:
        clf.save(join(MODEL_DIR, f"{model_name}.pkl"))

    return {"lang": lang, "model_name": model_name, "accuracy": accuracy, "macro_f1": macro_f1}


def plot_results(results: list[dict], classes: int, save_path: str | None = None) -> None:
    """Bar chart of accuracy and macro-F1 per language."""
    import matplotlib.pyplot as plt
    import numpy as np

    langs = [r["lang"].upper() for r in results]
    accs = [r["accuracy"] for r in results]
    f1s = [r["macro_f1"] for r in results]

    x = np.arange(len(langs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(langs) * 1.2), 4))
    bars_acc = ax.bar(x - width / 2, accs, width, label="Accuracy", color="steelblue")
    bars_f1 = ax.bar(x + width / 2, f1s, width, label="Macro F1", color="darkorange")

    for bars in (bars_acc, bars_f1):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(langs)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(f"COSC {classes}-class classifier — accuracy and macro F1 per language")
    ax.legend()
    ax.axhline(sum(accs) / len(accs), color="steelblue", linestyle="--",
               linewidth=0.8, alpha=0.6)
    ax.axhline(sum(f1s) / len(f1s), color="darkorange", linestyle="--",
               linewidth=0.8, alpha=0.6)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved chart: {save_path}")
    else:
        plt.show()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train COSC classifiers for all languages")
    parser.add_argument("--lang", help="Train a single language only")
    parser.add_argument(
        "--classes", type=int, default=52, choices=[6, 52],
        help="Number of classes (default: 52)"
    )
    parser.add_argument("--no-onnx", action="store_true", help="Save .pkl instead of .onnx")
    parser.add_argument("--plot", action="store_true", help="Plot accuracy/F1 after training")
    parser.add_argument(
        "--save-plot",
        metavar="PATH",
        help="Save accuracy/F1 chart to this path (implies --plot)"
    )
    args = parser.parse_args()

    langs = [args.lang] if args.lang else list(LANG_CONFIG)
    print(f"Training {len(langs)} language(s), {args.classes}-class models")

    results = []
    for lang in langs:
        r = train_language(lang, args.classes, not args.no_onnx)
        if r:
            results.append(r)

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"  {'Lang':<6} {'Model':<35} {'Accuracy':>10} {'Macro F1':>10}")
    print(f"  {'-'*6} {'-'*35} {'-'*10} {'-'*10}")
    for r in results:
        print(f"  {r['lang'].upper():<6} {r['model_name']:<35} {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f}")

    if results and (args.plot or args.save_plot):
        save_path = args.save_plot or join(REPORTS_DIR, f"cosc_{args.classes}class_all_langs.png")
        plot_results(results, args.classes, save_path=save_path if args.save_plot else None)
        if args.plot and not args.save_plot:
            plot_results(results, args.classes)


if __name__ == "__main__":
    main()
