#!/usr/bin/env python3
"""Check total feature count per language after FeatureUnion."""

import os
from os.path import dirname, join

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion

from train.classifiers import LinearSVCClassifier
from train.lang.feature_extractors import LanguageFeatureTransformer
from train.train_sentence_type_categorical import load_sentence_type_data

DATA_DIR = join(dirname(__file__), "clean_data")

LANGS = {
    "en": "sentence_types_EN.txt",
    "es": "sentence_types_ES.txt",
    "fr": "sentence_types_FR.txt",
    "de": "sentence_types_DE.txt",
    "ca": "sentence_types_CA.txt",
    "it": "sentence_types_IT.txt",
    "nl": "sentence_types_NL.txt",
    "pt": "sentence_types_PT.txt",
}

print("=" * 70)
print("Feature Count Analysis — All Languages")
print("=" * 70)

for lang, filename in LANGS.items():
    path = join(DATA_DIR, filename)
    texts, labels = load_sentence_type_data(path)
    print(f"\n{lang.upper()}: {len(texts)} samples")

    # Use LinearSVCClassifier to get the actual pipeline (with new TF-IDF params)
    categorical = LanguageFeatureTransformer(lang=lang, sparse=True)
    clf = LinearSVCClassifier(categorical_transformer=categorical)

    # Get the pipeline
    pipeline = clf.pipeline

    # Extract feature counts
    features_step = pipeline[0]  # ("features", FeatureUnion or TfidfVectorizer)
    features = features_step[1]

    features.fit(texts)

    # Count TF-IDF features
    if isinstance(features, FeatureUnion):
        tfidf = features.named_transformers["tfidf"]
        tfidf_vocab_size = len(tfidf.vocabulary_)

        categorical = features.named_transformers["categorical"]
        categorical_feature_names = categorical.get_feature_names_out()
        categorical_count = len(categorical_feature_names)
    else:
        tfidf_vocab_size = len(features.vocabulary_)
        categorical_count = 0

    total_count = tfidf_vocab_size + categorical_count

    status = "✓ OK" if total_count <= 100 else "✗ EXCEED"
    print(f"  TF-IDF vocabulary: {tfidf_vocab_size}")
    print(f"  Categorical features: {categorical_count}")
    print(f"  Total: {total_count} {status}")

print("\n" + "=" * 70)
print("Note: Feature count may vary slightly depending on train/test split")
print("Expected: TF-IDF 180 features + 14 categorical = 194 total (<=200 limit)")
print("=" * 70)
