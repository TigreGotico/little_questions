#!/usr/bin/env python3
"""Benchmark yes/no answer-polarity ONNX classifiers.

Evaluates the trained ONNX models on the held-out test set from
``TigreGotico/yes-no-multilingual``.  Saves text reports and plots.

Usage::

    python -m train.benchmark_yesno                    # all trained languages + multilingual
    python -m train.benchmark_yesno --lang en          # single language
    python -m train.benchmark_yesno --no-plot          # skip plots
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

MODEL_DIR = os.path.expanduser("~/.local/share/little_questions/yesno")
BUNDLED_DIR = join(dirname(dirname(__file__)), "little_questions", "models", "yesno")
REPORTS_DIR = join(dirname(__file__), "reports", "yesno")
VERSION = "0.9.0"
LABELS = ["yes", "no", "maybe"]

ALL_LANGUAGES = [
    "an","ar","bg","ca","cs","da","de","el","en","es","et","eu","fa","fi","fil",
    "fr","gl","he","hr","hu","id","is","it","ja","ko","lt","lv","ms","nb","nl",
    "nn","pl","pt","ro","ru","sk","sl","sv","th","tr","uk","vi","zh",
]


# ---------------------------------------------------------------------------
# ONNX inference wrapper
# ---------------------------------------------------------------------------

class YesNoOnnxScorer:
    def __init__(self, onnx_path: str) -> None:
        import onnxruntime as rt
        self.name = Path(onnx_path).stem
        self._sess = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self._input_name = self._sess.get_inputs()[0].name
        meta = self._sess.get_modelmeta().custom_metadata_map
        self.classes: list[str] = json.loads(meta.get("classes", "[]")) or LABELS

    def predict(self, text: str) -> str:
        result = self._sess.run(None, {self._input_name: [text]})
        label = result[0][0]
        return str(label) if not isinstance(label, (int, np.integer)) else self.classes[int(label)]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(scorer: YesNoOnnxScorer, x: list[str], y: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

    y_pred = [scorer.predict(t) for t in x]
    acc = accuracy_score(y, y_pred)
    mf1 = f1_score(y, y_pred, average="macro", zero_division=0)
    wf1 = f1_score(y, y_pred, average="weighted", zero_division=0)
    report = classification_report(y, y_pred, labels=LABELS, zero_division=0)
    cm = confusion_matrix(y, y_pred, labels=LABELS)
    per_class = {
        lbl: float(f1_score([yi == lbl for yi in y], [yp == lbl for yp in y_pred],
                             average="binary", zero_division=0))
        for lbl in LABELS
    }
    return {
        "accuracy": acc, "macro_f1": mf1, "weighted_f1": wf1,
        "report": report, "confusion_matrix": cm.tolist(),
        "labels": LABELS, "per_class_f1": per_class,
    }


def _find_model(lang: str) -> str | None:
    lang_upper = lang.upper()
    candidates = [
        join(BUNDLED_DIR, f"yesno_svm_cal_{lang_upper}_{VERSION}.onnx"),
        join(MODEL_DIR, f"yesno_svm_cal_{lang_upper}_{VERSION}.onnx"),
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None


def _find_multilingual_model() -> str | None:
    candidates = [
        join(BUNDLED_DIR, f"yesno_svm_cal_multilingual_{VERSION}.onnx"),
        join(MODEL_DIR, f"yesno_svm_cal_multilingual_{VERSION}.onnx"),
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None


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

    fig, ax = plt.subplots(figsize=(max(10, len(langs) * 0.6), 5))
    bars1 = ax.bar(x - width / 2, accs, width, label="Accuracy", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, mf1s, width, label="Macro F1", color="#DD8452")
    ax.set_xlabel("Language")
    ax.set_ylabel("Score")
    ax.set_title("Yes/No Answer-Polarity Classifier — All Languages")
    ax.set_xticks(x)
    ax.set_xticklabels([l.upper() for l in langs], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.yaxis.grid(True, alpha=0.3)
    fig.tight_layout()
    path = join(out_dir, "benchmark_yesno_overview.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion(result: dict, label: str, out_dir: str) -> str:
    import matplotlib.pyplot as plt

    cm = np.array(result["confusion_matrix"])
    labels = result["labels"]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Yes/No Confusion — {label}")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if cm_norm[i, j] > 0.6 else "black")
    fig.tight_layout()
    path = join(out_dir, f"benchmark_yesno_{label.lower()}_confusion.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def benchmark(langs: list[str] | None, plot: bool = True) -> None:
    from train.load_yesno import load_yesno_hf

    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    # Per-language models
    target_langs = langs or ALL_LANGUAGES
    for lang in target_langs:
        model_path = _find_model(lang)
        if not model_path:
            continue
        try:
            x, y = load_yesno_hf(lang=lang)
        except Exception as exc:
            print(f"[{lang.upper()}] data load error: {exc}")
            continue
        if not x:
            continue
        scorer = YesNoOnnxScorer(model_path)
        result = evaluate_model(scorer, x, y)
        results[lang] = result
        print(f"[{lang.upper():<4}]  accuracy={result['accuracy']:.4f}  "
              f"macro_f1={result['macro_f1']:.4f}  n={len(x)}")
        txt_path = join(REPORTS_DIR, f"yesno_svm_cal_{lang.upper()}_{VERSION}.txt")
        Path(txt_path).write_text(result["report"], encoding="utf-8")
        json_path = join(REPORTS_DIR, f"yesno_svm_cal_{lang.upper()}_{VERSION}_benchmark.json")
        Path(json_path).write_text(
            json.dumps({k: v for k, v in result.items() if k != "report"}, indent=2),
            encoding="utf-8",
        )

    # Multilingual model
    multi_path = _find_multilingual_model()
    if multi_path and not langs:
        try:
            x_all, y_all = load_yesno_hf(lang=None)
            scorer = YesNoOnnxScorer(multi_path)
            result = evaluate_model(scorer, x_all, y_all)
            results["multilingual"] = result
            print(f"[MULTI]  accuracy={result['accuracy']:.4f}  "
                  f"macro_f1={result['macro_f1']:.4f}  n={len(x_all)}")
            txt_path = join(REPORTS_DIR, f"yesno_svm_cal_multilingual_{VERSION}.txt")
            Path(txt_path).write_text(result["report"], encoding="utf-8")
            json_path = join(REPORTS_DIR, f"yesno_svm_cal_multilingual_{VERSION}_benchmark.json")
            Path(json_path).write_text(
                json.dumps({k: v for k, v in result.items() if k != "report"}, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"[MULTI] error: {exc}")

    if not results:
        print("No results — no models or data found.")
        return

    print(f"\n{'='*60}")
    print(f"  {'Lang':<14} {'Accuracy':>10} {'Macro F1':>10}")
    print(f"  {'-'*14} {'-'*10} {'-'*10}")
    for lang, r in results.items():
        print(f"  {lang.upper():<14} {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f}")

    if plot:
        print(f"\nSaving plots to {REPORTS_DIR}/")
        p = plot_overview(results, REPORTS_DIR)
        print(f"  {p}")
        if "multilingual" in results:
            p = plot_confusion(results["multilingual"], "multilingual", REPORTS_DIR)
            print(f"  {p}")
        if "en" in results:
            p = plot_confusion(results["en"], "en", REPORTS_DIR)
            print(f"  {p}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark yes/no ONNX classifiers")
    parser.add_argument("--lang", default=None, help="Single language code")
    parser.add_argument("--no-plot", dest="plot", action="store_false", default=True)
    args = parser.parse_args()

    langs = [args.lang] if args.lang else None
    benchmark(langs, plot=args.plot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
