#!/usr/bin/env python3
"""Compare different classifiers for COSC question classification."""

import argparse
import os
import sys
from os.path import join, dirname

sys.path.insert(0, join(dirname(__file__), ".."))

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from xdg import BaseDirectory as XDG

from train.classifiers import (
    LinearSVCClassifier,
    LogRegClassifier,
    RandomForestClassifier,
    NaiveBayesClassifier,
)

DATA_DIR = join(dirname(__file__), "clean_data")
MODEL_DIR = XDG.save_data_path("little_questions")

LANG_CONFIG = {
    "en": {"dataset": "raw_questions_0.7.0a1.txt", "suffix": "EN"},
    "es": {"dataset": "raw_questions_ES_googtx0.7.0a1.txt", "suffix": "ES"},
    "ca": {"dataset": "raw_questions_CA_apertiumtx0.7.0a1.txt", "suffix": "CA"},
    "pt": {"dataset": "raw_questions_PT_googtx0.7.0a1.txt", "suffix": "PT"},
    "fr": {"dataset": "raw_questions_FR_googtx0.7.0a1.txt", "suffix": "FR"},
    "de": {"dataset": "raw_questions_DE_0.8.0.txt", "suffix": "DE"},
    "it": {"dataset": "raw_questions_IT_0.8.0.txt", "suffix": "IT"},
    "nl": {"dataset": "raw_questions_NL_0.8.0.txt", "suffix": "NL"},
}

CLASSIFIERS = {
    "svm": LinearSVCClassifier,
    "logreg": LogRegClassifier,
    "rf": RandomForestClassifier,
    "nb": NaiveBayesClassifier,
}


def load_data(dataset_path, classes=6):
    x, y = [], []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            label = parts[0]
            if classes == 6:
                label = label.split(":")[0]
            x.append(parts[1])
            y.append(label)
    return x, y


def compare_classifiers(lang, classes=6):
    config = LANG_CONFIG.get(lang)
    if not config:
        return None

    dataset = config["dataset"]
    data_path = join(DATA_DIR, dataset)
    if not os.path.exists(data_path):
        return None

    x, y = load_data(data_path, classes)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.15, stratify=y, random_state=42
    )

    results = {}
    for name, clf_class in CLASSIFIERS.items():
        clf = clf_class(lang)
        clf.train(x_train, y_train)
        preds = clf.predict(x_test)

        from sklearn.metrics import accuracy_score

        acc = accuracy_score(y_test, preds)
        results[name] = acc
        print(f"  {name:10s}: {acc:.1%}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare classifiers")
    parser.add_argument("--lang", default="en", help="Language to test")
    parser.add_argument("--6", dest="classes6", action="store_true", help="Use 6-class")
    parser.add_argument("--all", action="store_true", help="Test all languages")
    args = parser.parse_args()

    classes = 6 if args.classes6 else 52

    if args.all:
        print(f"{'=' * 60}")
        print(f"Comparing classifiers for all languages ({classes}-class)")
        print(f"{'=' * 60}\n")

        all_results = {}
        for lang in LANG_CONFIG:
            print(f"\n{lang.upper()}:")
            results = compare_classifiers(lang, classes)
            if results:
                all_results[lang] = results

        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        print(f"\n{'Lang':<6}", end="")
        for clf in CLASSIFIERS:
            print(f"{clf:>8s}", end="")
        print(f"{'Best':>10s}")
        print("-" * 45)

        for lang, results in all_results.items():
            print(f"{lang:<6}", end="")
            best_clf = max(results, key=results.get)
            for clf, acc in results.items():
                marker = "*" if clf == best_clf else " "
                print(f"{marker}{acc:.1%}   ", end="")
            print(f"  {best_clf}")
    else:
        print(f"Comparing classifiers for {args.lang} ({classes}-class):\n")
        compare_classifiers(args.lang, classes)


if __name__ == "__main__":
    main()
