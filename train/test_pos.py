#!/usr/bin/env python3
"""Test if POS tagging helps COSC classification."""

import os
import sys
from os.path import join, dirname

sys.path.insert(0, join(dirname(__file__), ".."))

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

DATA_DIR = join(dirname(__file__), "clean_data")

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


class POSFeatures(BaseEstimator, TransformerMixin):
    """Extract POS tag n-grams as features."""

    def __init__(self, lang="en"):
        self.lang = lang
        self._pos_tagger = None

    def _get_tagger(self):
        if self._pos_tagger is None:
            if self.lang == "en":
                from nltk import pos_tag, word_tokenize

                self._pos_tag = lambda s: pos_tag(word_tokenize(s))
            elif self.lang in ["es", "pt", "ca", "fr", "de", "it", "nl"]:
                from brill_postaggers import BrillPostagger

                tagger = BrillPostagger.from_pretrained(self.lang)
                self._pos_tag = lambda s: tagger.tag(s.lower().split())
            else:
                from nltk import pos_tag, word_tokenize

                self._pos_tag = lambda s: pos_tag(word_tokenize(s))
        return self._pos_tag

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **kwargs):
        pos_tag = self._get_tagger()
        result = []
        for text in X:
            try:
                tags = pos_tag(text)
                tag_str = " ".join([t for _, t in tags])
                result.append(tag_str)
            except:
                result.append("")
        return result


def test_pos(lang, classes=6):
    x, y = load_data(lang, classes)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.15, stratify=y, random_state=42
    )

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
    baseline = accuracy_score(y_test, pipe.predict(x_test))
    print(f"    {baseline:.1%}")

    print("  TF-IDF word12 + POS tags...")
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
                            "pos",
                            Pipeline(
                                [
                                    ("pos", POSFeatures(lang)),
                                    (
                                        "tfidf",
                                        TfidfVectorizer(
                                            ngram_range=(1, 3), min_df=1, max_df=0.4
                                        ),
                                    ),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            ("clf", LinearSVC()),
        ]
    )
    pipe.fit(x_train, y_train)
    with_pos = accuracy_score(y_test, pipe.predict(x_test))
    print(f"    {with_pos:.1%}")

    print("  TF-IDF word12+char34 (best so far)...")
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
    with_char = accuracy_score(y_test, pipe.predict(x_test))
    print(f"    {with_char:.1%}")

    print("  TF-IDF word12+char34 + POS...")
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
                        (
                            "pos",
                            Pipeline(
                                [
                                    ("pos", POSFeatures(lang)),
                                    (
                                        "tfidf",
                                        TfidfVectorizer(
                                            ngram_range=(1, 3), min_df=1, max_df=0.4
                                        ),
                                    ),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            ("clf", LinearSVC()),
        ]
    )
    pipe.fit(x_train, y_train)
    combo = accuracy_score(y_test, pipe.predict(x_test))
    print(f"    {combo:.1%}")

    return {
        "baseline": baseline,
        "with_pos": with_pos,
        "with_char": with_char,
        "combo": combo,
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
            try:
                all_results[lang] = test_pos(lang, classes)
            except Exception as e:
                print(f"\n{lang.upper()}: ERROR - {e}")
    else:
        all_results[args.lang] = test_pos(args.lang, classes)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        f"\n{'Lang':<6} {'base':>8} {'+pos':>8} {'+char':>8} {'combo':>8} {'Best':>10}"
    )
    print("-" * 55)
    for lang, results in all_results.items():
        best = max(results, key=results.get)
        print(
            f"{lang:<6} {results['baseline']:>7.1%} {results['with_pos']:>7.1%} "
            f"{results['with_char']:>7.1%} {results['combo']:>7.1%} {best:>10}"
        )


if __name__ == "__main__":
    main()
