#!/usr/bin/env python3
"""Evaluate sentence-type classifiers and baselines.

Runs the trained SentenceTypeClassifier alongside the rule-based baselines
on a labelled dataset and writes reports + confusion-matrix plots.

Usage::

    # Evaluate all scorers on English sentence_types_EN.txt
    python -m train.eval_sentence_type

    # Evaluate on a specific language
    python -m train.eval_sentence_type --lang es

    # Save plots to reports/; skip interactive display
    python -m train.eval_sentence_type --save --no-show

    # Compare all languages
    python -m train.eval_sentence_type --all-langs
"""

from __future__ import annotations

import argparse
import logging
import os
from os.path import dirname, join
from pathlib import Path

from train.metrics import compare, evaluate

LOG = logging.getLogger(__name__)

DATA_DIR = join(dirname(__file__), "clean_data")
REPORTS_DIR = join(dirname(__file__), "reports")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_sentence_type_data(dataset_path: str) -> tuple[list[str], list[str]]:
    """Load a sentence-type dataset.

    Expected format (one sample per line)::

        1: This is a question?
        2: Open the door.

    Where the leading digit maps to:  1=question 2=command 3=statement 4=exclamation 5=request.
    Also supports bare ``label: text`` format used in ``sentence_types_*.txt``.
    """
    LABEL_MAP = {
        "1": "question",
        "2": "command",
        "3": "statement",
        "4": "exclamation",
        "5": "request",
    }
    texts, labels = [], []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ": " not in line:
                continue
            raw_label, text = line.split(": ", 1)
            label = LABEL_MAP.get(raw_label.strip(), raw_label.strip().lower())
            texts.append(text.strip())
            labels.append(label)
    return texts, labels


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------

def _get_scorers(lang: str) -> list:
    from train.baselines import HeuristicScorer, PunctuationScorer
    from little_questions.sentence_type import SentenceTypeClassifier

    scorers = [
        PunctuationScorer(),
        SentenceTypeClassifier.get_instance(lang),
    ]
    if lang == "en":
        scorers.insert(1, HeuristicScorer())
    return scorers


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_lang(
    lang: str,
    save: bool = False,
    show: bool = True,
) -> None:
    """Run evaluation for a single language."""
    dataset_path = join(DATA_DIR, f"sentence_types_{lang.upper()}.txt")
    if not os.path.exists(dataset_path):
        LOG.warning("Dataset not found for lang=%s: %s", lang, dataset_path)
        return

    texts, labels = load_sentence_type_data(dataset_path)
    if not texts:
        LOG.warning("No samples loaded from %s", dataset_path)
        return

    print(f"\n{'=' * 70}")
    print(f"Language: {lang.upper()}  |  {len(texts)} samples  |  {dataset_path}")
    print("=" * 70)

    scorers = _get_scorers(lang)
    results = []

    for scorer in scorers:
        result = evaluate(
            scorer,
            texts,
            labels,
            dataset=dataset_path,
            lang=lang,
        )
        print(f"\n{result}")
        results.append(result)

        if save:
            name = getattr(scorer, "name", type(scorer).__name__)
            prefix = f"{REPORTS_DIR}/sentence_type_{lang}_{name}"
            result.save(f"{prefix}.json")
            result.plot_confusion_matrix(
                normalise=True,
                save_path=f"{prefix}_cm.png",
            )
            result.plot_per_class_bars(
                metric="f1-score",
                save_path=f"{prefix}_f1.png",
            )
        elif show:
            result.plot_confusion_matrix(normalise=True)
            result.plot_per_class_bars(metric="f1-score")

    compare(
        results,
        save_path=f"{REPORTS_DIR}/sentence_type_{lang}_comparison.png" if save else None,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Evaluate sentence-type scorers against labelled datasets"
    )
    parser.add_argument("--lang", default="en", help="Language code (default: en)")
    parser.add_argument(
        "--all-langs",
        action="store_true",
        help="Evaluate all available languages",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help=f"Save reports and plots to {REPORTS_DIR}/",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip interactive plot display (useful in CI)",
    )
    args = parser.parse_args()

    if args.all_langs:
        from little_questions.constants import SUPPORTED_LANGUAGES
        langs = SUPPORTED_LANGUAGES
    else:
        langs = [args.lang]

    os.makedirs(REPORTS_DIR, exist_ok=True)

    for lang in langs:
        evaluate_lang(lang, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
