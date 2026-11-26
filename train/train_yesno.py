#!/usr/bin/env python3
"""Train yes/no answer polarity classifiers and export to ONNX.

Two model types:
  - ``svm_cal``: Platt-calibrated LinearSVC — for languages with sufficient
    written-text data; exports to ONNX via skl2onnx.
  - ``m2v``: Model2Vec potion-multilingual-128M + LinearSVC — for the
    multilingual case; saved as joblib artefacts (not ONNX).

With only 200 samples per language the best single-language option is a very
lightweight TF-IDF SVC.  The recommended deployment model is the multilingual
m2v variant trained on all languages combined.

Usage::

    python -m train.train_yesno                    # all languages + multilingual
    python -m train.train_yesno --lang en          # single language
    python -m train.train_yesno --type svm_cal     # svm_cal only
    python -m train.train_yesno --plot             # save plots after training
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from os.path import dirname, join
from pathlib import Path

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import train.mlflow_config as mlflow_config
from train.load_yesno import load_yesno_hf, LABELS

LOG = logging.getLogger(__name__)

HF_DATASET = "TigreGotico/yes-no-multilingual"
REPORTS_DIR = join(dirname(__file__), "reports", "yesno")
MODEL_DIR = os.path.expanduser("~/.local/share/little_questions/yesno")
VERSION = "0.9.0"

ALL_LANGUAGES = [
    "an","ar","bg","ca","cs","da","de","el","en","es","et","eu","fa","fi","fil",
    "fr","gl","he","hr","hu","id","is","it","ja","ko","lt","lv","ms","nb","nl",
    "nn","pl","pt","ro","ru","sk","sl","sv","th","tr","uk","vi","zh",
]


def _build_svm_cal_pipeline() -> Pipeline:
    tfidf = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        max_features=5_000,
    )
    clf = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000), cv=3, method="sigmoid")
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def _save_onnx(pipeline: Pipeline, path: str, classes: list[str]) -> None:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import StringTensorType
    import onnx

    initial_type = [("input", StringTensorType([None]))]
    options = {CalibratedClassifierCV: {"zipmap": False}}
    onnx_model = convert_sklearn(pipeline, initial_types=initial_type, options=options)
    if isinstance(onnx_model, tuple):
        onnx_model = onnx_model[0]

    meta = onnx_model.metadata_props.add()
    meta.key = "classes"
    meta.value = json.dumps(classes)

    flag = onnx_model.metadata_props.add()
    flag.key = "calibrated"
    flag.value = "true"

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(onnx_model, path)


def train_svm_cal(lang: str, x: list[str], y: list[str]) -> dict:
    lang_upper = lang.upper()
    model_name = f"yesno_svm_cal_{lang_upper}_{VERSION}"
    print(f"\n{'=' * 55}")
    print(f"Training  {model_name}  ({len(x)} samples)")
    print("=" * 55)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    pipe = _build_svm_cal_pipeline()
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

    onnx_path = join(MODEL_DIR, f"{model_name}.onnx")
    print(f"Exporting ONNX → {onnx_path}")
    _save_onnx(pipe, onnx_path, LABELS)

    import mlflow
    mlflow.set_experiment("yesno")
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({"model_type": "svm_cal", "lang": lang, "version": VERSION})
        mlflow.log_metrics({"accuracy": accuracy, "macro_f1": macro_f1, "weighted_f1": weighted_f1})
        mlflow.log_artifact(report_path)
        if os.path.exists(onnx_path):
            mlflow.log_artifact(onnx_path, artifact_path="onnx")

    return {"model_name": model_name, "lang": lang, "accuracy": accuracy,
            "macro_f1": macro_f1, "weighted_f1": weighted_f1}


def train_svm_cal_multilingual(x: list[str], y: list[str]) -> dict:
    """Train a single multilingual TF-IDF SVM on all languages combined.

    Character n-grams are language-agnostic and work well for the shallow
    lexical patterns in yes/no responses (the vocabulary is extremely small
    and stable across languages).
    """
    model_name = f"yesno_svm_cal_multilingual_{VERSION}"
    print(f"\n{'=' * 55}")
    print(f"Training  {model_name}  ({len(x)} samples, all languages)")
    print("=" * 55)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.15, stratify=y, random_state=42
    )

    pipe = _build_svm_cal_pipeline()
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

    onnx_path = join(MODEL_DIR, f"{model_name}.onnx")
    print(f"Exporting ONNX → {onnx_path}")
    _save_onnx(pipe, onnx_path, LABELS)

    import mlflow
    mlflow.set_experiment("yesno")
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({"model_type": "svm_cal_multilingual", "version": VERSION,
                           "n_languages": len(ALL_LANGUAGES)})
        mlflow.log_metrics({"accuracy": accuracy, "macro_f1": macro_f1, "weighted_f1": weighted_f1})
        mlflow.log_artifact(report_path)
        if os.path.exists(onnx_path):
            mlflow.log_artifact(onnx_path, artifact_path="onnx")

    return {"model_name": model_name, "lang": "multilingual", "accuracy": accuracy,
            "macro_f1": macro_f1, "weighted_f1": weighted_f1}


def train_m2v_multilingual(x: list[str], y: list[str]) -> dict:
    from train.classifiers import Model2VecClassifier

    model_name = f"yesno_m2v_multilingual_{VERSION}"
    print(f"\n{'=' * 55}")
    print(f"Training  {model_name}  ({len(x)} samples, all languages)")
    print("=" * 55)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.15, stratify=y, random_state=42
    )

    clf = Model2VecClassifier(
        model_name="minishlab/potion-multilingual-128M",
        use_tfidf_fusion=True,
    )
    clf.train(x_train, y_train)
    y_pred = clf.predict(x_test)

    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = join(REPORTS_DIR, f"{model_name}.txt")
    Path(report_path).write_text(report, encoding="utf-8")

    save_dir = join(MODEL_DIR, model_name)
    print(f"Saving model → {save_dir}")
    clf.save(save_dir)

    import mlflow
    mlflow.set_experiment("yesno")
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({"model_type": "m2v_multilingual", "version": VERSION,
                           "m2v_model": "potion-multilingual-128M", "tfidf_fusion": True})
        mlflow.log_metrics({"accuracy": accuracy, "macro_f1": macro_f1, "weighted_f1": weighted_f1})
        mlflow.log_artifact(report_path)

    return {"model_name": model_name, "lang": "multilingual", "accuracy": accuracy,
            "macro_f1": macro_f1, "weighted_f1": weighted_f1}


def plot_results(results: list[dict]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    names = [r["model_name"].replace(f"_{VERSION}", "") for r in results]
    accs = [r["accuracy"] for r in results]
    f1s = [r["macro_f1"] for r in results]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.4), 4))
    ax.bar(x - width / 2, accs, width, label="Accuracy", color="steelblue")
    ax.bar(x + width / 2, f1s, width, label="Macro F1", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Yes/No polarity classifiers")
    ax.legend()
    fig.tight_layout()

    path = join(REPORTS_DIR, "benchmark_yesno_overview.png")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved chart: {path}")
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    mlflow_config.setup()

    parser = argparse.ArgumentParser(description="Train yes/no polarity classifiers")
    parser.add_argument("--lang", default=None,
                        help="Single language code (default: all languages)")
    parser.add_argument("--type", choices=["svm_cal", "svm_cal_multilingual", "m2v"], default=None,
                        help="Model type to train (default: all three)")
    parser.add_argument("--plot", action="store_true", help="Save plots after training")
    args = parser.parse_args()

    results: list[dict] = []
    model_types = [args.type] if args.type else ["svm_cal", "svm_cal_multilingual", "m2v"]

    if "svm_cal" in model_types:
        langs = [args.lang] if args.lang else ALL_LANGUAGES
        for lang in langs:
            print(f"\nLoading yes/no data for lang={lang}")
            x, y = load_yesno_hf(lang=lang)
            print(f"  {len(x)} samples")
            if len(x) < 30:
                print(f"  Skipping {lang} — too few samples")
                continue
            r = train_svm_cal(lang, x, y)
            results.append(r)

    if ("svm_cal_multilingual" in model_types or "svm_cal" in model_types) and not args.lang:
        print("\nLoading full multilingual yes/no dataset for svm_cal_multilingual")
        x_all, y_all = load_yesno_hf(lang=None)
        print(f"  {len(x_all)} samples total")
        r = train_svm_cal_multilingual(x_all, y_all)
        results.append(r)

    if "m2v" in model_types and not args.lang:
        print("\nLoading full multilingual yes/no dataset")
        x_all, y_all = load_yesno_hf(lang=None)
        print(f"  {len(x_all)} samples total")
        r = train_m2v_multilingual(x_all, y_all)
        results.append(r)

    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    print(f"  {'Model':<45} {'Lang':>6} {'Accuracy':>10} {'Macro F1':>10}")
    print(f"  {'-'*45} {'-'*6} {'-'*10} {'-'*10}")
    for r in results:
        print(f"  {r['model_name']:<45} {r['lang']:>6} {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f}")

    if args.plot and results:
        plot_results(results)


if __name__ == "__main__":
    main()
