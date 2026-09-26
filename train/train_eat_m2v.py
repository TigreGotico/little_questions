#!/usr/bin/env python3
"""Train Model2Vec (potion-base) classifiers on EAT and save as joblib directories.

Trains 12 variants: {potion-base-2M, 8M, 32M} × {plain, +tfidf} × {53-class, 7-class}.

Usage::

    python -m train.train_eat_m2v                       # all 12 variants
    python -m train.train_eat_m2v --classes 7           # 7-class only (6 variants)
    python -m train.train_eat_m2v --model potion-base-8M  # one backbone, both class sizes
"""

from __future__ import annotations

import argparse
import logging
import os
from os.path import dirname, join
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

import train.mlflow_config as mlflow_config
from train.classifiers import Model2VecClassifier
from train.load_eat import load_eat, load_eat_hf

LOG = logging.getLogger(__name__)

HF_DATASET = "TigreGotico/EAT"
REPORTS_DIR = join(dirname(__file__), "reports", "eat")
MODEL_BASE_DIR = os.path.expanduser("~/.local/share/little_questions/eat_m2v")
VERSION = "0.9.0"

M2V_MODELS = [
    "minishlab/potion-base-2M",
    "minishlab/potion-base-8M",
    "minishlab/potion-base-32M",
]


def train_variant(
    model_name: str,
    use_tfidf: bool,
    classes: int,
    x_train: list[str],
    x_test: list[str],
    y_train: list[str],
    y_test: list[str],
) -> dict:
    clf = Model2VecClassifier(model_name=model_name, use_tfidf_fusion=use_tfidf)
    short = model_name.split("/")[-1]
    suffix = "+tfidf" if use_tfidf else ""
    variant_name = f"m2v-{short}{suffix}-en-{classes}c"

    print(f"\n{'=' * 60}")
    print(f"Training  {variant_name}")
    print("=" * 60)

    clf.train(x_train, y_train)

    y_pred = clf.predict(x_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = join(REPORTS_DIR, f"{variant_name}.txt")
    Path(report_path).write_text(report, encoding="utf-8")

    out_dir = join(MODEL_BASE_DIR, variant_name)
    print(f"Saving → {out_dir}")
    clf.save(out_dir)

    import mlflow
    mlflow.set_experiment("little-questions-eat")
    with mlflow.start_run(run_name=variant_name):
        mlflow.log_params({
            "model_name": model_name,
            "use_tfidf_fusion": use_tfidf,
            "n_classes": classes,
            "variant_name": variant_name,
            "test_size": 0.15,
            "version": VERSION,
        })
        mlflow.log_metrics({
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        })
        mlflow.log_artifact(report_path)

    return {
        "variant_name": variant_name,
        "model_name": model_name,
        "use_tfidf": use_tfidf,
        "classes": classes,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    mlflow_config.setup()

    parser = argparse.ArgumentParser(description="Train M2V EAT classifiers")
    parser.add_argument("--dataset", default=None,
                        help="Path to local EAT.tsv (default: load from HuggingFace)")
    parser.add_argument("--classes", type=int, choices=[7, 53],
                        help="Train only this class granularity (default: both)")
    parser.add_argument("--model", choices=[m.split("/")[-1] for m in M2V_MODELS],
                        help="Train a single backbone only")
    args = parser.parse_args()

    class_variants = [args.classes] if args.classes else [53, 7]
    backbone_filter = args.model

    results: list[dict] = []

    for n_cls in class_variants:
        if args.dataset:
            print(f"\nLoading EAT dataset ({n_cls}-class) from {args.dataset}")
            x, y = load_eat(args.dataset, classes=n_cls)
        else:
            print(f"\nLoading EAT dataset ({n_cls}-class) from HuggingFace ({HF_DATASET})")
            x, y = load_eat_hf(classes=n_cls)
        print(f"  {len(x)} samples  |  {len(set(y))} classes")

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.15, stratify=y, random_state=42
        )

        for m2v_model in M2V_MODELS:
            short = m2v_model.split("/")[-1]
            if backbone_filter and short != backbone_filter:
                continue
            for use_tfidf in (False, True):
                r = train_variant(m2v_model, use_tfidf, n_cls,
                                  x_train, x_test, y_train, y_test)
                results.append(r)

    print(f"\n{'=' * 70}")
    print("M2V Summary")
    print(f"{'=' * 70}")
    print(f"  {'Variant':<45} {'Classes':>8} {'Accuracy':>10} {'Macro F1':>10}")
    print(f"  {'-'*45} {'-'*8} {'-'*10} {'-'*10}")
    for r in results:
        print(f"  {r['variant_name']:<45} {r['classes']:>8} "
              f"{r['accuracy']:>10.4f} {r['macro_f1']:>10.4f}")


if __name__ == "__main__":
    main()
