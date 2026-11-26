#!/usr/bin/env python3
"""Evaluate and prune categorical features based on mutual information.

Computes MI for each categorical feature on the EN dataset and identifies
low-signal features (MI < 0.01 bits) for pruning.

Usage:
    python -m train.eval_sentence_type_categorical
"""

from __future__ import annotations

import logging
import os
from os.path import dirname, join

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import numpy as np

from train.lang.feature_extractors import LanguageFeatureExtractor_EN

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DATA_DIR = join(dirname(__file__), "clean_data")


def load_sentence_type_data(path: str) -> tuple[list[str], list[str]]:
    """Load sentence-type dataset."""
    texts: list[str] = []
    labels: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            label, text = parts
            texts.append(text)
            labels.append(label)
    return texts, labels


def compute_feature_importance():
    """Compute mutual information for each EN categorical feature."""
    print("=" * 70)
    print("Feature Importance Analysis — EN Sentence-Type Classifier")
    print("=" * 70)

    # Load data
    data_path = join(DATA_DIR, "sentence_types_EN.txt")
    texts, labels = load_sentence_type_data(data_path)
    print(f"Loaded {len(texts)} samples, {len(set(labels))} classes")

    # Extract features
    extractor = LanguageFeatureExtractor_EN()
    feature_list = []
    feature_names = None

    print("\nExtracting categorical features...")
    for i, text in enumerate(texts):
        feat_dict = extractor.extract(text)
        if feature_names is None:
            feature_names = list(feat_dict.keys())
        feature_list.append([feat_dict.get(name, 0.0) for name in feature_names])
        if (i + 1) % 1000 == 0:
            print(f"  Extracted {i + 1} samples...")

    X = np.array(feature_list)  # shape: (n_samples, n_features)
    print(f"Feature matrix shape: {X.shape}")

    # Encode labels to integers
    le = LabelEncoder()
    y = le.fit_transform(labels)
    print(f"Classes: {list(le.classes_)}")

    # Compute mutual information
    print("\nComputing mutual information...")
    mi_scores = mutual_info_classif(X, y, random_state=42)

    # Sort by MI and display
    feature_mi = list(zip(feature_names, mi_scores))
    feature_mi.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 70)
    print(f"{'Feature':<30} {'MI (bits)':<15} {'Status':<20}")
    print("=" * 70)

    high_signal = []
    low_signal = []
    for name, mi in feature_mi:
        status = "KEEP" if mi >= 0.01 else "PRUNE"
        if mi >= 0.01:
            high_signal.append(name)
        else:
            low_signal.append(name)
        print(f"{name:<30} {mi:<15.4f} {status:<20}")

    print("=" * 70)
    print(f"\nHigh-signal features (MI >= 0.01): {len(high_signal)}")
    print(f"  {high_signal}")
    print(f"\nLow-signal features (MI < 0.01): {len(low_signal)}")
    print(f"  {low_signal}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    mi_array = np.array([mi for _, mi in feature_mi])
    print(f"Total features: {len(mi_scores)}")
    print(f"Mean MI: {mi_array.mean():.4f} bits")
    print(f"Median MI: {np.median(mi_array):.4f} bits")
    print(f"Min MI: {mi_array.min():.4f} bits")
    print(f"Max MI: {mi_array.max():.4f} bits")
    print(f"Std MI: {mi_array.std():.4f} bits")

    return feature_mi, high_signal, low_signal


if __name__ == "__main__":
    compute_feature_importance()
