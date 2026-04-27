#!/usr/bin/env python3
"""Train EAT classifiers and export to ONNX.

All models are fully exportable via skl2onnx — inference requires only onnxruntime.

Usage::

    python -m train.train_eat                    # all baselines, 53-class + 7-class
    python -m train.train_eat --classes 7        # 7-class only
    python -m train.train_eat --model svm        # single model type
    python -m train.train_eat --no-onnx          # save .pkl instead
    python -m train.train_eat --plot             # save benchmark bar chart after training
"""

from __future__ import annotations

import argparse
import logging
import os
from os.path import dirname, join
from pathlib import Path

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier as _SGD
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import train.mlflow_config as mlflow_config
from train.classifiers import CalibratedLinearSVCClassifier
from train.load_eat import load_eat, load_eat_hf

LOG = logging.getLogger(__name__)

HF_DATASET = "TigreGotico/EAT"
REPORTS_DIR = join(dirname(__file__), "reports", "eat")
MODEL_DIR = os.path.expanduser("~/.local/share/little_questions/eat")
VERSION = "0.9.0"

# ---------------------------------------------------------------------------
# Baseline pipeline factories — all produce plain sklearn Pipelines that
# skl2onnx can convert to ONNX without custom transformers.
# ---------------------------------------------------------------------------

def _word_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.9, sublinear_tf=True)


def _char_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1,
                           max_df=0.9, sublinear_tf=True)


BASELINES: dict[str, tuple[str, callable]] = {
    "svm":     ("word", lambda: LinearSVC(C=1.0, max_iter=2000)),
    "logreg":  ("word", lambda: LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0)),
    "sgd":     ("word", lambda: _SGD(loss="hinge", penalty="l2", alpha=1e-3, random_state=42)),
    "svm_cal": None,  # handled specially via CalibratedLinearSVCClassifier
}


def build_pipeline(model_type: str) -> Pipeline:
    vec_kind, clf_factory = BASELINES[model_type]
    tfidf = _char_tfidf() if vec_kind == "char" else _word_tfidf()
    return Pipeline([("tfidf", tfidf), ("clf", clf_factory())])


# ---------------------------------------------------------------------------
# ONNX export helper (no custom transformers, so this always works)
# ---------------------------------------------------------------------------

def save_onnx(pipeline: Pipeline, path: str, classes: list[str], calibrated: bool = False) -> None:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import StringTensorType
    import onnx, json

    initial_type = [("input", StringTensorType([None]))]
    if calibrated:
        options = {CalibratedClassifierCV: {"zipmap": False}}
    else:
        options = {LinearSVC: {"nocl": True}}
    onnx_model = convert_sklearn(pipeline, initial_types=initial_type, options=options)
    if isinstance(onnx_model, tuple):
        onnx_model = onnx_model[0]

    meta = onnx_model.metadata_props.add()
    meta.key = "classes"
    meta.value = json.dumps(classes)

    if calibrated:
        flag = onnx_model.metadata_props.add()
        flag.key = "calibrated"
        flag.value = "true"

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(onnx_model, path)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model_type: str,
    classes: int,
    x_train: list[str],
    x_test: list[str],
    y_train: list[str],
    y_test: list[str],
) -> dict:
    n_class_tag = f"eat{classes}"
    # char_svm gets a different tag
    model_name = f"{n_class_tag}_{model_type}_EN_{VERSION}"
    print(f"\n{'=' * 60}")
    print(f"Training  {model_name}")
    print("=" * 60)

    is_calibrated = model_type == "svm_cal"
    if is_calibrated:
        clf_wrapper = CalibratedLinearSVCClassifier()
        clf_wrapper.train(x_train, y_train)
        pipe = clf_wrapper.clf
    else:
        pipe = build_pipeline(model_type)
        pipe.fit(x_train, y_train)

    y_pred = list(pipe.predict(x_test))
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = join(REPORTS_DIR, f"{model_name}.txt")
    Path(report_path).write_text(report, encoding="utf-8")

    import mlflow

    onnx_path = join(MODEL_DIR, f"{model_name}.onnx")
    print(f"Exporting ONNX → {onnx_path}")
    label_list = sorted(set(y_train))
    save_onnx(pipe, onnx_path, label_list, calibrated=is_calibrated)  # raises on failure — no pkl fallback

    mlflow.set_experiment(mlflow_config.EXPERIMENT_COSC.replace("cosc", "eat"))
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({
            "model_type": model_type,
            "n_classes": classes,
            "model_name": model_name,
            "test_size": 0.15,
            "version": VERSION,
        })
        mlflow.log_metrics({
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        })
        mlflow.log_artifact(report_path)
        if os.path.exists(onnx_path):
            mlflow.log_artifact(onnx_path, artifact_path="onnx")

    return {
        "model_name": model_name,
        "model_type": model_type,
        "classes": classes,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_overview(results: list[dict], save_path: str | None = None) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    for n_cls in (53, 7):
        subset = [r for r in results if r["classes"] == n_cls]
        if not subset:
            continue
        names = [r["model_type"] for r in subset]
        accs = [r["accuracy"] for r in subset]
        f1s = [r["macro_f1"] for r in subset]

        x = np.arange(len(names))
        width = 0.35
        fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.4), 4))
        bars_a = ax.bar(x - width / 2, accs, width, label="Accuracy", color="steelblue")
        bars_f = ax.bar(x + width / 2, f1s, width, label="Macro F1", color="darkorange")
        for bars in (bars_a, bars_f):
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title(f"EAT {n_cls}-class baselines — accuracy & macro F1")
        ax.legend()
        fig.tight_layout()

        path = save_path or join(REPORTS_DIR, f"benchmark_eat{n_cls}_overview.png")
        if n_cls != 53 and save_path:
            path = save_path.replace(".png", f"_{n_cls}class.png")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved chart: {path}")
        plt.close(fig)


def plot_comparison(results: list[dict], save_path: str | None = None) -> None:
    """Line plot: macro F1 per model type for 53-class vs 7-class."""
    import matplotlib.pyplot as plt

    model_types = list(dict.fromkeys(r["model_type"] for r in results))
    f1_53 = {r["model_type"]: r["macro_f1"] for r in results if r["classes"] == 53}
    f1_7 = {r["model_type"]: r["macro_f1"] for r in results if r["classes"] == 7}

    fig, ax = plt.subplots(figsize=(max(6, len(model_types) * 1.4), 4))
    if f1_53:
        ax.plot(model_types, [f1_53.get(m, 0) for m in model_types],
                marker="o", label="53-class", color="steelblue")
    if f1_7:
        ax.plot(model_types, [f1_7.get(m, 0) for m in model_types],
                marker="s", label="7-class", color="darkorange")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro F1")
    ax.set_title("EAT — macro F1 by model type and granularity")
    ax.legend()
    fig.tight_layout()

    path = save_path or join(REPORTS_DIR, "benchmark_eat_model_comparison.png")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved chart: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    mlflow_config.setup()

    parser = argparse.ArgumentParser(description="Train EAT classifiers")
    parser.add_argument("--dataset", default=None,
                        help="Path to local EAT.tsv (default: load from HuggingFace TigreGotico/EAT)")
    parser.add_argument("--classes", type=int, choices=[7, 53],
                        help="Train only this class granularity (default: both)")
    parser.add_argument("--model", choices=[k for k, v in BASELINES.items()],
                        help="Train a single model type only")
    parser.add_argument("--plot", action="store_true", help="Save benchmark plots after training")
    args = parser.parse_args()

    class_variants = [args.classes] if args.classes else [53, 7]
    model_types = [args.model] if args.model else list(BASELINES)

    results: list[dict] = []

    for n_cls in class_variants:
        if args.dataset:
            print(f"\nLoading EAT dataset ({n_cls}-class) from {args.dataset}")
            x, y = load_eat(args.dataset, classes=n_cls)
        else:
            print(f"\nLoading EAT dataset ({n_cls}-class) from HuggingFace ({HF_DATASET})")
            x, y = load_eat_hf(classes=n_cls)
        print(f"  {len(x)} samples  |  {len(set(y))} classes")

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.15, stratify=y, random_state=42
        )

        for mt in model_types:
            r = train_model(mt, n_cls, x_train, x_test, y_train, y_test)
            results.append(r)

    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    print(f"  {'Model':<35} {'Classes':>8} {'Accuracy':>10} {'Macro F1':>10}")
    print(f"  {'-'*35} {'-'*8} {'-'*10} {'-'*10}")
    for r in results:
        print(f"  {r['model_name']:<35} {r['classes']:>8} {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f}")

    if args.plot and results:
        plot_overview(results)
        plot_comparison(results)


if __name__ == "__main__":
    main()
