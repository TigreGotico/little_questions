#!/usr/bin/env python3
"""Feature engineering experiments for COSC classification."""

import argparse
import os
import sys
from os.path import join, dirname

sys.path.insert(0, join(dirname(__file__), ".."))

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_DIR = join(dirname(__file__), "clean_data")


def load_data(lang="en", classes=6):
    suffix_map = {
        "en": "raw_questions_0.7.0a1.txt",
        "es": "raw_questions_ES_googtx0.7.0a1.txt",
        "ca": "raw_questions_CA_apertiumtx0.7.0a1.txt",
        "pt": "raw_questions_PT_googtx0.7.0a1.txt",
        "fr": "raw_questions_FR_googtx0.7.0a1.txt",
        "de": "raw_questions_DE_0.8.0.txt",
        "it": "raw_questions_IT_0.8.0.txt",
        "nl": "raw_questions_NL_0.8.0.txt",
    }
    path = join(DATA_DIR, suffix_map.get(lang, "raw_questions_0.7.0a1.txt"))

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


FEATURES = {
    "tfidf_word12": lambda: TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=1, max_df=0.4
    ),
    "tfidf_word13": lambda: TfidfVectorizer(
        analyzer="word", ngram_range=(1, 3), min_df=1, max_df=0.4
    ),
    "tfidf_char34": lambda: TfidfVectorizer(
        analyzer="char", ngram_range=(3, 4), min_df=1, max_df=0.4
    ),
    "tfidf_char45": lambda: TfidfVectorizer(
        analyzer="char_wb", ngram_range=(4, 5), min_df=1, max_df=0.4
    ),
    "tfidf_word12_char34": lambda: FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word", ngram_range=(1, 2), min_df=1, max_df=0.4
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char", ngram_range=(3, 4), min_df=1, max_df=0.4
                ),
            ),
        ]
    ),
    "tfidf_word12_char45": lambda: FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word", ngram_range=(1, 2), min_df=1, max_df=0.4
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(4, 5), min_df=1, max_df=0.4
                ),
            ),
        ]
    ),
    "tfidf_word13_char34": lambda: FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word", ngram_range=(1, 3), min_df=1, max_df=0.4
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char", ngram_range=(3, 4), min_df=1, max_df=0.4
                ),
            ),
        ]
    ),
    "tfidf_word13_char45": lambda: FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word", ngram_range=(1, 3), min_df=1, max_df=0.4
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(4, 5), min_df=1, max_df=0.4
                ),
            ),
        ]
    ),
}


def test_features(lang, classes=6):
    x, y = load_data(lang, classes)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.15, stratify=y, random_state=42
    )

    results = {}
    for name, vec_fn in FEATURES.items():
        try:
            vec = vec_fn()
            pipe = Pipeline([("features", vec), ("clf", LinearSVC())])
            pipe.fit(x_train, y_train)
            preds = pipe.predict(x_test)
            acc = accuracy_score(y_test, preds)
            results[name] = acc
            print(f"  {name:25s}: {acc:.1%}")
        except Exception as e:
            print(f"  {name:25s}: ERROR - {e}")
            results[name] = 0

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare feature engineering approaches"
    )
    parser.add_argument("--lang", default="en", help="Language to test")
    parser.add_argument("--6", dest="classes6", action="store_true", help="Use 6-class")
    parser.add_argument("--all", action="store_true", help="Test all languages")
    args = parser.parse_args()

    classes = 6 if args.classes6 else 52

    if args.all:
        all_results = {}
        for lang in ["en", "es", "pt", "fr", "de", "it", "nl"]:
            print(f"\n{lang.upper()} ({classes}-class):")
            results = test_features(lang, classes)
            all_results[lang] = results

        print("\n" + "=" * 60)
        print("BEST FEATURE PER LANGUAGE")
        print("=" * 60)
        for lang, results in all_results.items():
            best = max(results, key=results.get)
            print(f"  {lang}: {best} ({results[best]:.1%})")
    else:
        print(f"Feature engineering comparison for {args.lang} ({classes}-class):\n")
        test_features(args.lang, classes)


if __name__ == "__main__":
    main()
