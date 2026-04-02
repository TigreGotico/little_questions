#!/usr/bin/env python3
"""Validate exported pickle models match sklearn predictions.

Tests that each exported .pkl model makes identical predictions
to the training pipeline on 100 random test samples per language.

Usage:
    python -m train.validate_exported_models
"""

from __future__ import annotations

import logging
import os
from os.path import dirname, join
from xdg import BaseDirectory as XDG
import joblib
import numpy as np

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


def validate_language(lang: str) -> dict | None:
    """Validate exported model for *lang*.

    Trains fresh pipeline, exports to pickle, loads it,
    and checks predictions match on 100 random samples.

    Returns a dict with ``lang``, ``match_count``, ``total`` on success.
    """
    config = LANG_CONFIG.get(lang)
    if not config:
        LOG.error("Unknown language: %s", lang)
        return None

    suffix = config["suffix"]
    model_name = f"sentence_type_{suffix}_0.8.0"

    print(f"\n{'=' * 70}")
    print(f"Validating {lang.upper()} — {model_name}")
    print("=" * 70)

    data_path = join(DATA_DIR, config["dataset"])
    if not os.path.exists(data_path):
        LOG.warning("Dataset not found: %s", data_path)
        return None

    # Load and split data
    x, y = load_sentence_type_data(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.15, stratify=y, random_state=42
    )

    # Train fresh pipeline
    categorical_transformer = LanguageFeatureTransformer(lang=lang, sparse=True)
    clf_sklearn = LinearSVCClassifier(categorical_transformer=categorical_transformer)
    clf_sklearn.train(x_train, y_train)

    # Load exported pickle model
    pkl_path = join(MODEL_DIR, f"{model_name}.pkl")
    if not os.path.exists(pkl_path):
        LOG.warning("Pickle model not found: %s", pkl_path)
        return None

    clf_pkl = joblib.load(pkl_path)
    print(f"Loaded model from {pkl_path}")

    # Test on 100 random samples from test set (or all if fewer than 100)
    sample_count = min(100, len(x_test))
    sample_indices = np.random.choice(len(x_test), sample_count, replace=False)
    x_sample = [x_test[i] for i in sample_indices]

    # Get predictions from both
    pred_sklearn = clf_sklearn.predict(x_sample)
    pred_pkl = clf_pkl.predict(x_sample)

    # Compare
    matches = sum(1 for p_sk, p_pkl in zip(pred_sklearn, pred_pkl) if p_sk == p_pkl)
    match_pct = 100.0 * matches / sample_count

    print(f"Tested on {sample_count} samples")
    print(f"Matches: {matches}/{sample_count} ({match_pct:.1f}%)")

    if match_pct < 100.0:
        print(f"⚠ WARNING: {sample_count - matches} mismatches detected")
        for i, (sk, pkl) in enumerate(zip(pred_sklearn, pred_pkl)):
            if sk != pkl:
                print(f"  Sample {i}: sklearn={sk}, pkl={pkl}")

    status = "✓ PASS" if match_pct == 100.0 else "✗ FAIL"
    print(f"Result: {status}")

    return {
        "lang": lang,
        "match_count": matches,
        "total": sample_count,
        "match_pct": match_pct,
    }


def main():
    print("=" * 70)
    print("Model Validation — Pickle vs Sklearn Predictions")
    print("=" * 70)

    results = []
    for lang in LANG_CONFIG.keys():
        result = validate_language(lang)
        if result:
            results.append(result)

    # Summary
    if results:
        print(f"\n{'=' * 70}")
        print("Validation Summary")
        print("=" * 70)
        print(f"{'Language':<10} {'Matches':<15} {'Success Rate':<15}")
        print("-" * 70)
        for r in results:
            pct_str = f"{r['match_pct']:.1f}%"
            print(f"{r['lang'].upper():<10} {r['match_count']}/{r['total']:<10} {pct_str:<15}")

        all_pass = all(r["match_pct"] == 100.0 for r in results)
        if all_pass:
            print("\n✓ All models validated successfully")
        else:
            print("\n✗ Some models have mismatches")


if __name__ == "__main__":
    main()
