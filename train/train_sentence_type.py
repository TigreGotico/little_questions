#!/usr/bin/env python3
"""Train sentence-type classifiers for all languages and export to ONNX.

Each language produces a single ONNX model file saved to the XDG data directory.
The model is a Pipeline(FeatureUnion([TF-IDF word, TF-IDF char]) → LinearSVC).

Usage::

    python -m train.train_sentence_type               # All languages
    python -m train.train_sentence_type --lang en     # One language only
    python -m train.train_sentence_type --no-onnx     # Save .pkl instead
"""

from __future__ import annotations

import argparse
import logging
import os
from os.path import dirname, join
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from xdg import BaseDirectory as XDG

import train.mlflow_config as mlflow_config

LOG = logging.getLogger(__name__)

DATA_DIR = join(dirname(__file__), "clean_data")
REPORTS_DIR = join(dirname(__file__), "reports")
MODEL_DIR = XDG.save_data_path("little_questions")

LANGS = ["en", "es", "pt", "ca", "fr", "de", "it", "nl"]
SUFFIX = {lang: lang.upper() for lang in LANGS}

VALID_LABELS = frozenset({"question", "command", "statement", "exclamation", "request"})


def load_data(path: str) -> tuple[list[str], list[str]]:
    """Load sentence-type dataset.

    Supports two formats:
    - ``N: label text`` — numeric prefix, label as first word after colon
    - ``label: text``   — label directly before the colon

    In both cases the label must be one of the five valid sentence types.
    """
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or ": " not in line:
                continue
            prefix, rest = line.split(": ", 1)
            prefix = prefix.strip()
            if prefix in VALID_LABELS:
                # ``label: text``
                label, text = prefix, rest.strip()
            else:
                # ``N: [N] label text`` — skip leading numeric tokens, take first valid label
                parts = rest.split(None)
                label, text = None, None
                for i, token in enumerate(parts):
                    if token in VALID_LABELS:
                        label = token
                        text = " ".join(parts[i + 1:]).strip()
                        break
                if not label or not text:
                    continue
            texts.append(text)
            labels.append(label)
    return texts, labels


def make_pipeline() -> Pipeline:
    """Return a TF-IDF + LinearSVC pipeline exportable to ONNX.

    Uses word unigrams + bigrams (same configuration as the COSC classifiers)
    which is known to export cleanly via skl2onnx.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                  min_df=1, max_df=0.9, sublinear_tf=True)),
        ("clf", LinearSVC(C=1.0, max_iter=2000)),
    ])


def export_onnx(pipe: Pipeline, path: str) -> None:
    """Export a fitted Pipeline to ONNX, embedding class names as metadata."""
    import json
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import StringTensorType
    import onnx

    initial_type = [("input", StringTensorType([None]))]
    onnx_model = convert_sklearn(
        pipe,
        initial_types=initial_type,
        options={LinearSVC: {"nocl": True}},
    )
    if isinstance(onnx_model, tuple):
        onnx_model = onnx_model[0]

    # Embed class labels as model metadata so inference can map indices → labels.
    classes = list(pipe.named_steps["clf"].classes_)
    meta = onnx_model.metadata_props.add()
    meta.key = "classes"
    meta.value = json.dumps(classes)

    onnx.save_model(onnx_model, path)


def train_language(lang: str, export_onnx_flag: bool = True) -> dict | None:
    """Train and export the sentence-type classifier for *lang*.

    Returns a result dict or None when the dataset is missing.
    """
    sfx = SUFFIX[lang]
    model_name = f"sentence_type_{sfx}_0.8.0"
    dataset_path = join(DATA_DIR, f"sentence_types_{sfx}.txt")

    if not os.path.exists(dataset_path):
        LOG.warning("Dataset not found: %s", dataset_path)
        return None

    print(f"\n{'=' * 60}")
    print(f"Training sentence-type  {lang.upper()} — {model_name}")
    print("=" * 60)

    texts, labels = load_data(dataset_path)
    print(f"Loaded {len(texts)} samples  |  classes: {sorted(set(labels))}")

    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.15, stratify=labels, random_state=42
    )

    pipe = make_pipeline()
    pipe.fit(x_train, y_train)

    y_pred = pipe.predict(x_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    print(f"\n{report}")

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = join(REPORTS_DIR, f"{model_name}.txt")
    Path(report_path).write_text(report, encoding="utf-8")

    import mlflow

    if export_onnx_flag:
        onnx_path = join(MODEL_DIR, f"{model_name}.onnx")
        print(f"Exporting ONNX → {onnx_path}")
        try:
            export_onnx(pipe, onnx_path)
        except Exception as exc:
            LOG.warning("ONNX export failed (%s); saving pkl", exc)
            onnx_path = None
            import joblib
            joblib.dump(pipe, join(MODEL_DIR, f"{model_name}.pkl"))
    else:
        onnx_path = None
        import joblib
        joblib.dump(pipe, join(MODEL_DIR, f"{model_name}.pkl"))

    mlflow.set_experiment(mlflow_config.EXPERIMENT_COSC)
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({
            "lang": lang, "model_name": model_name,
            "model_type": "tfidf-svm-sentence-type",
            "features": "word(1,4)", "test_size": 0.15,
        })
        mlflow.log_metrics({
            "accuracy": accuracy, "macro_f1": macro_f1, "weighted_f1": weighted_f1,
        })
        mlflow.log_artifact(report_path)
        if onnx_path and os.path.exists(onnx_path):
            mlflow.log_artifact(onnx_path, artifact_path="onnx")

    return {"lang": lang, "model_name": model_name, "accuracy": accuracy, "macro_f1": macro_f1}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    mlflow_config.setup()

    parser = argparse.ArgumentParser(description="Train sentence-type classifiers")
    parser.add_argument("--lang", help="Train a single language only")
    parser.add_argument("--no-onnx", action="store_true", help="Save .pkl instead of .onnx")
    args = parser.parse_args()

    langs = [args.lang] if args.lang else LANGS
    print(f"Training {len(langs)} language(s)")

    results = []
    for lang in langs:
        r = train_language(lang, not args.no_onnx)
        if r:
            results.append(r)

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"  {'Lang':<6} {'Model':<35} {'Accuracy':>10} {'Macro F1':>10}")
    print(f"  {'-'*6} {'-'*35} {'-'*10} {'-'*10}")
    for r in results:
        print(f"  {r['lang'].upper():<6} {r['model_name']:<35} {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f}")


if __name__ == "__main__":
    main()
