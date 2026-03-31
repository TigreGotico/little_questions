#!/usr/bin/env python3
"""Train the English COSC question classifier.

This is a convenience wrapper around :mod:`train.train_all` for the EN language.
Prefer ``python -m train.train_all --lang en`` for batch workflows.

Usage::

    python -m train.train_en                # Train 52-class model (default)
    python -m train.train_en --classes 6    # Train 6-class model
    python -m train.train_en --no-onnx      # Save as .pkl instead
    python -m train.train_en --plot         # Plot accuracy/F1 after training
"""

from __future__ import annotations

import argparse
import logging

from train.train_all import train_language, plot_results

LOG = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train English COSC classifier")
    parser.add_argument(
        "--classes", type=int, default=52, choices=[6, 52],
        help="Number of classes (default: 52)",
    )
    parser.add_argument("--no-onnx", action="store_true", help="Save .pkl instead of .onnx")
    parser.add_argument("--plot", action="store_true", help="Plot accuracy/F1 after training")
    parser.add_argument(
        "--dataset",
        default="raw_questions_EN_balanced_0.8.0.txt",
        help="Dataset filename inside train/clean_data/",
    )
    args = parser.parse_args()

    result = train_language("en", args.classes, not args.no_onnx)
    if result and args.plot:
        plot_results([result], args.classes)


if __name__ == "__main__":
    main()
