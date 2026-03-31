#!/usr/bin/env python3
"""Test model2vec with local multilingual model."""

import os
import sys
from os.path import join, dirname

sys.path.insert(0, join(dirname(__file__), ".."))

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline

DATA_DIR = join(dirname(__file__), "clean_data")
MODEL_PATH = "/home/miro/PycharmProjects/Machine Learning Workspace/notebooks/m2v/ovos-intents-multilingual-potion-multilingual-128M"

LANG_DATA = {
    "en": "raw_questions_0.7.0a1.txt",
    "es": "raw_questions_ES_googtx0.7.0a1.txt",
    "pt": "raw_questions_PT_googtx0.7.0a1.txt",
    "fr": "raw_questions_FR_googtx0.7.0a1.txt",
    "de": "raw_questions_DE_0.8.0.txt",
    "it": "raw_questions_IT_0.8.0.txt",
    "nl": "raw_questions_NL_0.8.0.txt",
}


def load_data(lang, classes=6):
    path = join(DATA_DIR, LANG_DATA.get(lang, LANG_DATA["en"]))
    x, y = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) != 2:
                continue
            label = parts[0].split(":")[0] if classes == 6 else parts[0]
            x.append(parts[1])
            y.append(label)
    return x, y


def test_m2v(lang, classes=6):
    from model2vec import StaticModel
    import numpy as np

    x, y = load_data(lang, classes)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.15, stratify=y, random_state=42
    )

    print(f"\n{lang.upper()} ({classes}-class):")

    # Baseline TF-IDF
    print("  TF-IDF word12...")
    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word", ngram_range=(1, 2), min_df=1, max_df=0.4
                ),
            ),
            ("clf", LinearSVC()),
        ]
    )
    pipe.fit(x_train, y_train)
    tfidf_acc = accuracy_score(y_test, pipe.predict(x_test))
    print(f"    {tfidf_acc:.1%}")

    # TF-IDF + char
    print("  TF-IDF word12+char34...")
    pipe = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                analyzer="word",
                                ngram_range=(1, 2),
                                min_df=1,
                                max_df=0.4,
                            ),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char",
                                ngram_range=(3, 4),
                                min_df=1,
                                max_df=0.4,
                            ),
                        ),
                    ]
                ),
            ),
            ("clf", LinearSVC()),
        ]
    )
    pipe.fit(x_train, y_train)
    tfidf_char_acc = accuracy_score(y_test, pipe.predict(x_test))
    print(f"    {tfidf_char_acc:.1%}")

    # model2vec
    print("  model2vec (potion-multilingual-128M)...")
    model = StaticModel.from_pretrained(MODEL_PATH)
    X_train_emb = model.encode(x_train)
    X_test_emb = model.encode(x_test)
    clf = LinearSVC()
    clf.fit(X_train_emb, y_train)
    m2v_acc = accuracy_score(y_test, clf.predict(X_test_emb))
    print(f"    {m2v_acc:.1%}")

    # TF-IDF + model2vec
    print("  TF-IDF + model2vec...")
    tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, max_df=0.4)
    X_train_tfidf = tfidf.fit_transform(x_train)
    X_test_tfidf = tfidf.transform(x_test)

    from scipy.sparse import hstack

    X_train_combo = hstack([X_train_tfidf, X_train_emb])
    X_test_combo = hstack([X_test_tfidf, X_test_emb])

    clf = LinearSVC()
    clf.fit(X_train_combo, y_train)
    combo_acc = accuracy_score(y_test, clf.predict(X_test_combo))
    print(f"    {combo_acc:.1%}")

    return {
        "tfidf": tfidf_acc,
        "tfidf_char": tfidf_char_acc,
        "m2v": m2v_acc,
        "combo": combo_acc,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="en")
    parser.add_argument("--6", dest="classes6", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    classes = 6 if args.classes6 else 52

    all_results = {}
    if args.all:
        for lang in ["en", "es", "de", "it", "nl"]:
            all_results[lang] = test_m2v(lang, classes)
    else:
        all_results[args.lang] = test_m2v(args.lang, classes)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        f"\n{'Lang':<6} {'tfidf':>8} {'+char':>8} {'m2v':>8} {'combo':>8} {'Best':>10}"
    )
    print("-" * 55)
    for lang, results in all_results.items():
        best = max(results, key=results.get)
        print(
            f"{lang:<6} {results['tfidf']:>7.1%} {results['tfidf_char']:>7.1%} "
            f"{results['m2v']:>7.1%} {results['combo']:>7.1%} {best:>10}"
        )


if __name__ == "__main__":
    main()
