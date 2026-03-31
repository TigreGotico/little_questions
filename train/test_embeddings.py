#!/usr/bin/env python3
"""Compare model2vec vs TF-IDF for COSC classification."""

import os
import sys
from os.path import join, dirname

sys.path.insert(0, join(dirname(__file__), ".."))

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

DATA_DIR = join(dirname(__file__), "clean_data")

LANG_DATA = {
    "en": "raw_questions_0.7.0a1.txt",
    "es": "raw_questions_ES_googtx0.7.0a1.txt",
    "ca": "raw_questions_CA_apertiumtx0.7.0a1.txt",
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


def compare_embeddings(lang, classes=6):
    x, y = load_data(lang, classes)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.15, stratify=y, random_state=42
    )

    results = {}

    print(f"\n{lang.upper()} ({classes}-class):")

    print("  TF-IDF word12 (baseline)...")
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
    results["tfidf_word12"] = accuracy_score(y_test, pipe.predict(x_test))
    print(f"    {results['tfidf_word12']:.1%}")

    print("  TF-IDF word12+char34 (best)...")
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
    results["tfidf_word12_char34"] = accuracy_score(y_test, pipe.predict(x_test))
    print(f"    {results['tfidf_word12_char34']:.1%}")

    print("  model2vec (fasttext)...")
    try:
        from model2vec import StaticModel

        model = StaticModel.from_pretrained(
            f"sentence-transformers/{lang}-MiniLM-L6-v2"
        )

        pipe = Pipeline([("embed", model), ("clf", LinearSVC())])
        pipe.fit(x_train, y_train)
        results["m2v_fasttext"] = accuracy_score(y_test, pipe.predict(x_test))
        print(f"    {results['m2v_fasttext']:.1%}")
    except Exception as e:
        print(f"    ERROR: {e}")
        results["m2v_fasttext"] = 0

    print("  model2vec (distiluse)...")
    try:
        from model2vec import StaticModel

        model = StaticModel.from_pretrained(
            "sentence-transformers/distiluse-base-multilingual"
        )

        pipe = Pipeline([("embed", model), ("clf", LinearSVC())])
        pipe.fit(x_train, y_train)
        results["m2v_distiluse"] = accuracy_score(y_test, pipe.predict(x_test))
        print(f"    {results['m2v_distiluse']:.1%}")
    except Exception as e:
        print(f"    ERROR: {e}")
        results["m2v_distiluse"] = 0

    return results


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
        for lang in LANG_DATA:
            all_results[lang] = compare_embeddings(lang, classes)
    else:
        all_results[args.lang] = compare_embeddings(args.lang, classes)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        f"\n{'Lang':<6} {'tfidf12':>10} {'tfidf+c':>10} {'m2v-ft':>10} {'m2v-dist':>10} {'Best':>12}"
    )
    print("-" * 65)

    for lang, results in all_results.items():
        best = max(results, key=results.get)
        print(
            f"{lang:<6} {results['tfidf_word12']:>9.1%} {results['tfidf_word12_char34']:>9.1%} "
            f"{results['m2v_fasttext']:>9.1%} {results['m2v_distiluse']:>9.1%} {best:>12}"
        )


if __name__ == "__main__":
    main()
