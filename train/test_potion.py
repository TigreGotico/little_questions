#!/usr/bin/env python3
"""Test Potion models from HuggingFace."""

import os
import sys
from os.path import join, dirname

sys.path.insert(0, join(dirname(__file__), ".."))

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

DATA_DIR = join(dirname(__file__), "clean_data")

MODELS = {
    "potion-base-2M": "minishlab/potion-base-2M",
    "potion-base-4M": "minishlab/potion-base-4M",
    "potion-base-8M": "minishlab/potion-base-8M",
    "potion-base-32M": "minishlab/potion-base-32M",
    "potion-multilingual-128M": "minishlab/potion-multilingual-128M",
}

LANG_DATA = {
    "en": ("en", "raw_questions_0.7.0a1.txt"),
    "es": ("es", "raw_questions_ES_googtx0.7.0a1.txt"),
    "de": ("de", "raw_questions_DE_0.8.0.txt"),
    "pt": ("pt", "raw_questions_PT_googtx0.7.0a1.txt"),
    "fr": ("fr", "raw_questions_FR_googtx0.7.0a1.txt"),
    "it": ("it", "raw_questions_IT_0.8.0.txt"),
    "nl": ("nl", "raw_questions_NL_0.8.0.txt"),
}


def load_data(lang, classes=6):
    suffix = LANG_DATA[lang][1]
    path = join(DATA_DIR, suffix)
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


def test_models(lang="en", classes=6):
    from model2vec import StaticModel

    x, y = load_data(lang, classes)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.15, stratify=y, random_state=42
    )

    print(f"\n{'=' * 60}")
    print(f"Testing on {lang.upper()} ({classes}-class)")
    print(f"{'=' * 60}")

    # Baseline TF-IDF
    print("\nTF-IDF word12+char34 (baseline)...")
    tfidf_word = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=1, max_df=0.4
    )
    tfidf_char = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 4), min_df=1, max_df=0.4
    )
    X_train_word = tfidf_word.fit_transform(x_train)
    X_test_word = tfidf_word.transform(x_test)
    X_train_char = tfidf_char.fit_transform(x_train)
    X_test_char = tfidf_char.transform(x_test)
    X_train_tfidf = hstack([X_train_word, X_train_char])
    X_test_tfidf = hstack([X_test_word, X_test_char])

    clf = LinearSVC()
    clf.fit(X_train_tfidf, y_train)
    baseline_acc = accuracy_score(y_test, clf.predict(X_test_tfidf))
    print(f"  {baseline_acc:.1%}")

    results = {"baseline": baseline_acc}

    for name, repo_id in MODELS.items():
        print(f"\n{name}...")
        try:
            model = StaticModel.from_pretrained(repo_id)
            print(f"  dim={model.dim}")

            # m2v alone
            X_train_emb = model.encode(x_train)
            X_test_emb = model.encode(x_test)
            clf = LinearSVC()
            clf.fit(X_train_emb, y_train)
            m2v_acc = accuracy_score(y_test, clf.predict(X_test_emb))
            print(f"  m2v alone: {m2v_acc:.1%}")

            # combo with TF-IDF
            X_train_combo = hstack([X_train_tfidf, X_train_emb])
            X_test_combo = hstack([X_test_tfidf, X_test_emb])
            clf = LinearSVC()
            clf.fit(X_train_combo, y_train)
            combo_acc = accuracy_score(y_test, clf.predict(X_test_combo))
            print(f"  +TF-IDF: {combo_acc:.1%}")

            results[name + "_alone"] = m2v_acc
            results[name + "_combo"] = combo_acc
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {k:35s}: {v:.1%}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="en")
    parser.add_argument("--6", dest="classes6", action="store_true")
    args = parser.parse_args()

    classes = 6 if args.classes6 else 52
    test_models(args.lang, classes)


if __name__ == "__main__":
    main()
