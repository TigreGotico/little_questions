#!/usr/bin/env python3
"""Train English COSC question classifier.

Usage:
    python -m train.train_en           # Train 52-class model
    python -m train.train_en --6       # Train 6-class model
    python -m train.train_en --onnx    # Export as ONNX
"""

import argparse
import os
from os.path import join, dirname

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from xdg import BaseDirectory as XDG

from train.classifiers import LinearSVCClassifier
from train.utils import load_data

DATA_DIR = join(dirname(__file__), "clean_data")
REPORTS_DIR = join(dirname(__file__), "reports")
MODEL_DIR = XDG.save_data_path("little_questions")


def train(lang, dataset, model_name, classes=52, export_onnx=True):
    """Train a model and optionally export as ONNX."""
    print(f"Training {model_name} ({classes}-class)...")

    data_path = join(DATA_DIR, dataset)
    if not os.path.exists(data_path):
        print(f"Dataset not found: {data_path}")
        print("Please add training data to train/clean_data/")
        return

    x, y = load_data(data_path, classes)
    print(f"Loaded {len(x)} samples")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.15, stratify=y, random_state=42
    )

    clf = LinearSVCClassifier(lang)
    clf.train(x_train, y_train)

    preds = clf.predict(x_test)
    report = classification_report(y_test, preds)
    print(f"\nClassification Report:\n{report}")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(join(REPORTS_DIR, f"{model_name}.txt"), "w") as f:
        f.write(report)

    if export_onnx:
        onnx_path = join(MODEL_DIR, f"{model_name}.onnx")
        print(f"\nExporting to ONNX: {onnx_path}")
        clf.save_onnx(onnx_path)
    else:
        pkl_path = join(MODEL_DIR, f"{model_name}.pkl")
        print(f"\nSaving as pickle: {pkl_path}")
        clf.save(pkl_path)

    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Train English COSC classifier")
    parser.add_argument(
        "--6", action="store_true", help="Train 6-class model instead of 52-class"
    )
    parser.add_argument("--no-onnx", action="store_true", help="Skip ONNX export")
    parser.add_argument(
        "--dataset", default="raw_questions_EN_0.8.0.txt", help="Dataset filename"
    )
    args = parser.parse_args()

    if args._6:
        model_name = "questions6_svm_EN_0.8.0"
        classes = 6
    else:
        model_name = "questions52_svm_EN_0.8.0"
        classes = 52

    train("en", args.dataset, model_name, classes, not args.no_onnx)


if __name__ == "__main__":
    main()
