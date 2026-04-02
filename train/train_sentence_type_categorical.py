#!/usr/bin/env python3
"""Train sentence-type classifiers with optional categorical features.

Tests both TF-IDF-only (baseline) and TF-IDF+categorical pipelines to measure
feature engineering impact.

Usage:
    python -m train.train_sentence_type_categorical --lang en
    python -m train.train_sentence_type_categorical --lang es --with-categorical
"""

from __future__ import annotations

import argparse
import logging
import os
from os.path import dirname, join
from pathlib import Path

from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from train.classifiers import LinearSVCClassifier
from train.lang.feature_extractors import LanguageFeatureTransformer

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DATA_DIR = join(dirname(__file__), "clean_data")
REPORTS_DIR = join(dirname(__file__), "reports")

LANG_CONFIG: dict[str, dict[str, str]] = {
    "en": {"dataset": "sentence_types_EN.txt", "suffix": "EN"},
    "es": {"dataset": "sentence_types_ES.txt", "suffix": "ES"},
    "ca": {"dataset": "sentence_types_CA.txt", "suffix": "CA"},
    "pt": {"dataset": "sentence_types_PT.txt", "suffix": "PT"},
    "fr": {"dataset": "sentence_types_FR.txt", "suffix": "FR"},
    "de": {"dataset": "sentence_types_DE.txt", "suffix": "DE"},
    "it": {"dataset": "sentence_types_IT.txt", "suffix": "IT"},
    "nl": {"dataset": "sentence_types_NL.txt", "suffix": "NL"},
}


def load_sentence_type_data(path: str) -> tuple[list[str], list[str]]:
    """Load sentence-type dataset.

    File format: one sample per line, ``LABEL: text`` or ``ID: LABEL text``

    Returns:
        Tuple of (texts, labels)
    """
    texts: list[str] = []
    labels: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Handle both "LABEL: text" and "ID: LABEL text" formats
            parts = line.split(": ", 1)
            if len(parts) != 2:
                continue

            left, right = parts
            # Check if left part is a digit (ID format)
            if left.isdigit():
                # Format: ID: LABEL text
                subparts = right.split(" ", 1)
                if len(subparts) == 2:
                    label, text = subparts
                else:
                    continue
            else:
                # Format: LABEL: text
                label, text = left, right

            texts.append(text)
            labels.append(label)
    return texts, labels


def train_language(
    lang: str,
    with_categorical: bool = False,
) -> dict | None:
    """Train a sentence-type classifier for *lang*.

    Returns a dict with ``lang``, ``model_name``, ``accuracy``, ``macro_f1``,
    ``categorical_improvement`` on success, or ``None`` when the dataset is missing.
    """
    config = LANG_CONFIG.get(lang)
    if not config:
        LOG.error("Unknown language: %s", lang)
        return None

    suffix = config["suffix"]
    mode_suffix = "categorical" if with_categorical else "tfidf"
    model_name = f"sentence_type_{suffix}_{mode_suffix}_0.8.0"

    print(f"\n{'=' * 70}")
    print(f"Training {lang.upper()} — {model_name}")
    print(f"Mode: {'TF-IDF + categorical features' if with_categorical else 'TF-IDF only (baseline)'}")
    print("=" * 70)

    data_path = join(DATA_DIR, config["dataset"])
    if not os.path.exists(data_path):
        LOG.warning("Dataset not found: %s", data_path)
        return None

    x, y = load_sentence_type_data(data_path)
    print(f"Loaded {len(x)} samples  |  {len(set(y))} classes")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.15, stratify=y, random_state=42
    )

    # Train model
    categorical_transformer = None
    if with_categorical:
        categorical_transformer = LanguageFeatureTransformer(
            lang=lang, sparse=True
        )

    clf = LinearSVCClassifier(categorical_transformer=categorical_transformer)
    clf.train(x_train, y_train)

    # Evaluate
    y_pred = clf.predict(x_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    print(f"\n{report}")

    from sklearn.metrics import accuracy_score, f1_score

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = join(REPORTS_DIR, f"{model_name}.txt")
    Path(report_path).write_text(report, encoding="utf-8")

    return {
        "lang": lang,
        "model_name": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train sentence-type classifiers with optional categorical features"
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="Language code (en, es, ca, fr, de, it, nl, pt)",
    )
    parser.add_argument(
        "--with-categorical",
        action="store_true",
        help="Use TF-IDF + categorical features (default: TF-IDF only)",
    )
    parser.add_argument(
        "--all-languages",
        action="store_true",
        help="Train all languages",
    )
    args = parser.parse_args()

    if args.all_languages:
        languages = list(LANG_CONFIG.keys())
    else:
        languages = [args.lang]

    results = []
    for lang in languages:
        result = train_language(lang, with_categorical=args.with_categorical)
        if result:
            results.append(result)

    # Summary
    if results:
        print(f"\n{'=' * 70}")
        print("Summary")
        print("=" * 70)
        print(f"{'Language':<10} {'Accuracy':<12} {'Macro F1':<12} {'Weighted F1':<12}")
        print("-" * 70)
        for r in results:
            print(
                f"{r['lang'].upper():<10} {r['accuracy']:<12.4f} {r['macro_f1']:<12.4f} {r['weighted_f1']:<12.4f}"
            )


if __name__ == "__main__":
    main()
