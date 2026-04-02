#!/usr/bin/env python3
"""Export sentence-type classifiers to pickle (.pkl) models (one per language).

Final pipelines include TF-IDF + categorical features:
- 194 total features per language (180 TF-IDF + 14 categorical, <=200 limit)
- Raw text input, predictions output
- Sklearn pipeline + joblib pickle format (production-ready)

Usage:
    python -m train.export_sentence_type_pkl --lang en
    python -m train.export_sentence_type_pkl --all-languages
"""

from __future__ import annotations

import argparse
import logging
import os
from os.path import dirname, join
from pathlib import Path
from xdg import BaseDirectory as XDG

from sklearn.model_selection import train_test_split

from train.classifiers import LinearSVCClassifier
from train.lang.feature_extractors import LanguageFeatureTransformer
from train.train_sentence_type_categorical import load_sentence_type_data

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DATA_DIR = join(dirname(__file__), "clean_data")
MODEL_DIR = XDG.save_data_path("little_questions")

LANG_CONFIG: dict[str, dict[str, str]] = {
    "en": {"dataset": "sentence_types_EN.txt", "suffix": "EN"},
    "es": {"dataset": "sentence_types_ES.txt", "suffix": "ES"},
    "fr": {"dataset": "sentence_types_FR.txt", "suffix": "FR"},
    "de": {"dataset": "sentence_types_DE.txt", "suffix": "DE"},
    "it": {"dataset": "sentence_types_IT.txt", "suffix": "IT"},
    "nl": {"dataset": "sentence_types_NL.txt", "suffix": "NL"},
    "pt": {"dataset": "sentence_types_PT.txt", "suffix": "PT"},
}


def export_language(lang: str) -> dict | None:
    """Train and export sentence-type classifier for *lang* to pickle.

    Returns a dict with ``lang``, ``model_name``, ``pkl_path`` on success.
    """
    config = LANG_CONFIG.get(lang)
    if not config:
        LOG.error("Unknown language: %s", lang)
        return None

    suffix = config["suffix"]
    model_name = f"sentence_type_{suffix}_0.8.0"

    print(f"\n{'=' * 70}")
    print(f"Exporting {lang.upper()} → {model_name}.pkl")
    print("=" * 70)

    data_path = join(DATA_DIR, config["dataset"])
    if not os.path.exists(data_path):
        LOG.warning("Dataset not found: %s", data_path)
        return None

    # Load and split data
    x, y = load_sentence_type_data(data_path)
    print(f"Loaded {len(x)} samples  |  {len(set(y))} classes")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.15, stratify=y, random_state=42
    )

    # Train classifier with categorical features
    categorical_transformer = LanguageFeatureTransformer(lang=lang, sparse=True)
    clf = LinearSVCClassifier(categorical_transformer=categorical_transformer)
    clf.train(x_train, y_train)

    # Export to pickle (joblib)
    os.makedirs(MODEL_DIR, exist_ok=True)
    pkl_path = join(MODEL_DIR, f"{model_name}.pkl")
    print(f"Saving to {pkl_path}...")

    try:
        clf.save(pkl_path)
        file_size_mb = os.path.getsize(pkl_path) / (1024 * 1024)
        print(f"✓ Pickle export successful ({file_size_mb:.2f} MB)")
    except Exception as exc:
        LOG.error("Pickle export failed: %s", exc)
        return None

    return {
        "lang": lang,
        "model_name": model_name,
        "pkl_path": pkl_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export sentence-type classifiers to ONNX"
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="Language code (en, es, ca, fr, de, it, nl, pt)",
    )
    parser.add_argument(
        "--all-languages",
        action="store_true",
        help="Export all languages",
    )
    args = parser.parse_args()

    if args.all_languages:
        languages = list(LANG_CONFIG.keys())
    else:
        languages = [args.lang]

    results = []
    for lang in languages:
        result = export_language(lang)
        if result:
            results.append(result)

    # Summary
    if results:
        print(f"\n{'=' * 70}")
        print("Pickle Export Summary")
        print("=" * 70)
        for r in results:
            print(f"{r['lang'].upper()}: {r['pkl_path']}")

        print("\n✓ All pickle models exported successfully")


if __name__ == "__main__":
    main()
