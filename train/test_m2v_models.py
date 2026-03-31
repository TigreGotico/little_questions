#!/usr/bin/env python3
"""Test model2vec with various monolingual and multilingual models."""

import os
import sys
from os.path import join, dirname

sys.path.insert(0, join(dirname(__file__), ".."))

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = join(dirname(__file__), "clean_data")

MODELS = {
    "multilingual": "/home/miro/PycharmProjects/Machine Learning Workspace/notebooks/m2v/ovos-intents-multilingual-potion-multilingual-128M",
    "serafim-100m": "/home/miro/PycharmProjects/Machine Learning Workspace/notebooks/m2v/ovos-intents-pt-PT-m2v-256-serafim-100m-portuguese-pt-sentence-encoder",
    "albertina-100m": "/home/miro/PycharmProjects/Machine Learning Workspace/notebooks/m2v/ovos-intents-pt-PT-m2v-256-albertina-100m-portuguese-ptbr-encoder",
    "serafim-335m": "/home/miro/PycharmProjects/Machine Learning Workspace/notebooks/m2v/ovos-intents-pt-PT-m2v-256-serafim-335m-portuguese-pt-sentence-encoder",
    "bert-base-pt": "/home/miro/PycharmProjects/Machine Learning Workspace/notebooks/m2v/ovos-intents-pt-PT-m2v-256-bert-base-portuguese-cased",
    "bert-large-pt": "/home/miro/PycharmProjects/Machine Learning Workspace/notebooks/m2v/ovos-intents-pt-PT-m2v-256-bert-large-portuguese-cased",
}


def load_data(lang, classes=6):
    path = join(DATA_DIR, f"raw_questions_{lang.upper()}_0.8.0.txt")
    if not os.path.exists(path):
        return None, None
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


def test_models(lang="pt", classes=6):
    from model2vec import StaticModel
    from scipy.sparse import hstack

    x, y = load_data(lang, classes)
    if not x:
        print(f"No data for {lang}")
        return None

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.15, stratify=y, random_state=42
    )

    print(f"\n{'=' * 60}")
    print(f"Testing on PORTUGUESE ({classes}-class)")
    print(f"{'=' * 60}")

    # Baseline
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

    for name, path in MODELS.items():
        if not os.path.exists(path):
            continue

        print(f"\n{name}...")
        try:
            model = StaticModel.from_pretrained(path)
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
        print(f"  {k:30s}: {v:.1%}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="pt")
    parser.add_argument("--6", dest="classes6", action="store_true")
    args = parser.parse_args()

    classes = 6 if args.classes6 else 52
    test_models(args.lang, classes)


if __name__ == "__main__":
    main()
