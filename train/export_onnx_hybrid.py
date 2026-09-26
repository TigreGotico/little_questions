#!/usr/bin/env python3
"""Export hybrid ONNX models: TF-IDF + LinearSVC only (categorical features computed externally).

HYBRID APPROACH FOR ONNX COMPLIANCE:
====================================
Since sklearn2onnx cannot convert custom feature transformers, this approach:
1. Creates ONNX models for TF-IDF + LinearSVC only
2. Categorical features are computed via sklearn preprocessing (outside ONNX)
3. Both are combined at inference time for end-to-end predictions

This maintains the accuracy benefits of categorical features while providing
ONNX-compatible core model for deployment.

Usage:
    python -m train.export_onnx_hybrid --lang en
    python -m train.export_onnx_hybrid --all-languages
"""

from __future__ import annotations

import argparse
import logging
import os
from os.path import dirname, join
from pathlib import Path
from xdg import BaseDirectory as XDG
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC as _LinearSVC

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
    """Export TF-IDF + LinearSVC pipeline to ONNX (without categorical features).

    Returns a dict with ``lang``, ``model_name``, ``onnx_path`` on success.
    """
    config = LANG_CONFIG.get(lang)
    if not config:
        LOG.error("Unknown language: %s", lang)
        return None

    suffix = config["suffix"]
    model_name = f"sentence_type_{suffix}_tfidf_svm_0.8.0"

    print(f"\n{'=' * 70}")
    print(f"Exporting {lang.upper()} (ONNX-only TF-IDF + LinearSVC)")
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

    # Train TF-IDF + LinearSVC only (no categorical features)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.4, max_features=180)),
        ("clf", _LinearSVC()),
    ])
    pipeline.fit(x_train, y_train)

    # Export to ONNX
    os.makedirs(MODEL_DIR, exist_ok=True)
    onnx_path = join(MODEL_DIR, f"{model_name}.onnx")
    print(f"Exporting to {onnx_path}...")

    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import StringTensorType

        initial_type = [("input", StringTensorType([None]))]
        onnx_model = convert_sklearn(
            pipeline, initial_types=initial_type, options={_LinearSVC: {"nocl": True}}
        )
        if isinstance(onnx_model, tuple):
            onnx_model = onnx_model[0]

        import onnx
        import json

        # Embed class labels as metadata
        classes = list(pipeline.named_steps["clf"].classes_)
        if classes:
            meta = onnx_model.metadata_props.add()
            meta.key = "classes"
            meta.value = json.dumps(classes)

        onnx.save_model(onnx_model, onnx_path)
        file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"✓ ONNX export successful ({file_size_mb:.2f} MB)")
        print(f"  NOTE: Categorical features NOT included in ONNX.")
        print(f"  Use pickle models for full accuracy with categorical features.")
    except Exception as exc:
        LOG.error("ONNX export failed: %s", exc)
        return None

    return {
        "lang": lang,
        "model_name": model_name,
        "onnx_path": onnx_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export TF-IDF + LinearSVC to ONNX (categorical features excluded)"
    )
    parser.add_argument("--lang", default="en")
    parser.add_argument("--all-languages", action="store_true")
    args = parser.parse_args()

    languages = list(LANG_CONFIG.keys()) if args.all_languages else [args.lang]

    results = []
    for lang in languages:
        result = export_language(lang)
        if result:
            results.append(result)

    if results:
        print(f"\n{'=' * 70}")
        print("ONNX Export Summary")
        print("=" * 70)
        for r in results:
            print(f"{r['lang'].upper()}: {r['onnx_path']}")

        print("\n⚠  WARNING: These ONNX models contain TF-IDF + LinearSVC only.")
        print("    For full accuracy (with categorical features), use pickle models.")


if __name__ == "__main__":
    main()
