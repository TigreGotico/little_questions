#!/usr/bin/env python3
"""Train COSC classifiers for all languages and export to ONNX."""

import argparse
import os
import sys
from os.path import join, dirname

sys.path.insert(0, join(dirname(__file__), ".."))

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from xdg import BaseDirectory as XDG

from train.classifiers import LinearSVCClassifier
from train.utils import load_data

DATA_DIR = join(dirname(__file__), "clean_data")
REPORTS_DIR = join(dirname(__file__), "reports")
MODEL_DIR = XDG.save_data_path("little_questions")

LANG_CONFIG = {
    "en": {
        "dataset": "raw_questions_0.7.0a1.txt",
        "suffix": "EN",
    },
    "es": {
        "dataset": "raw_questions_ES_googtx0.7.0a1.txt",
        "suffix": "ES",
    },
    "ca": {
        "dataset": "raw_questions_CA_apertiumtx0.7.0a1.txt",
        "suffix": "CA",
    },
    "pt": {
        "dataset": "raw_questions_PT_googtx0.7.0a1.txt",
        "suffix": "PT",
    },
    "fr": {
        "dataset": "raw_questions_FR_googtx0.7.0a1.txt",
        "suffix": "FR",
    },
    "de": {
        "dataset": "raw_questions_DE_0.8.0.txt",
        "suffix": "DE",
    },
    "it": {
        "dataset": "raw_questions_IT_0.8.0.txt",
        "suffix": "IT",
    },
    "nl": {
        "dataset": "raw_questions_NL_0.8.0.txt",
        "suffix": "NL",
    },
}


def train_language(lang, classes=52, export_onnx=True):
    """Train a model for a specific language."""
    config = LANG_CONFIG.get(lang)
    if not config:
        print(f"Unknown language: {lang}")
        return False

    dataset = config["dataset"]
    suffix = config["suffix"]

    if classes == 6:
        model_name = f"questions6_svm_{suffix}_0.8.0"
    else:
        model_name = f"questions52_svm_{suffix}_0.8.0"

    print(f"\n{'=' * 60}")
    print(f"Training {lang.upper()} ({model_name})")
    print(f"{'=' * 60}")

    data_path = join(DATA_DIR, dataset)
    if not os.path.exists(data_path):
        print(f"Dataset not found: {data_path}")
        return False

    x, y = load_data(data_path, classes)
    print(f"Loaded {len(x)} samples")
    print(f"Classes: {len(set(y))}")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.15, stratify=y, random_state=42
    )

    clf = LinearSVCClassifier(lang)
    clf.train(x_train, y_train)

    preds = clf.predict(x_test)
    report = classification_report(y_test, preds, zero_division=0)
    print(f"\nClassification Report:\n{report}")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(join(REPORTS_DIR, f"{model_name}.txt"), "w") as f:
        f.write(report)

    if export_onnx:
        onnx_path = join(MODEL_DIR, f"{model_name}.onnx")
        print(f"\nExporting to ONNX: {onnx_path}")
        try:
            clf.save_onnx(onnx_path)
            print(f"Model saved: {onnx_path}")
        except Exception as e:
            print(f"ONNX export failed: {e}")
            print("Saving as pickle instead...")
            pkl_path = join(MODEL_DIR, f"{model_name}.pkl")
            clf.save(pkl_path)
            print(f"Model saved: {pkl_path}")
    else:
        pkl_path = join(MODEL_DIR, f"{model_name}.pkl")
        print(f"\nSaving as pickle: {pkl_path}")
        clf.save(pkl_path)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Train COSC classifiers for all languages"
    )
    parser.add_argument("--lang", help="Train specific language only")
    parser.add_argument(
        "--6", dest="classes6", action="store_true", help="Train 6-class models"
    )
    parser.add_argument("--no-onnx", action="store_true", help="Skip ONNX export")
    args = parser.parse_args()

    classes = 6 if args.classes6 else 52

    if args.lang:
        langs = [args.lang]
    else:
        langs = list(LANG_CONFIG.keys())

    print(f"Training {len(langs)} languages with {classes}-class models")

    results = {}
    for lang in langs:
        results[lang] = train_language(lang, classes, not args.no_onnx)

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    for lang, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {lang.upper()}")


if __name__ == "__main__":
    main()
