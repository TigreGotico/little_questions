"""Shared utilities for train scripts."""

from __future__ import annotations


def load_data(dataset_path: str, classes: int = 52) -> tuple[list[str], list[str]]:
    """Load training data from a dataset file.

    File format: one sample per line, ``LABEL:subtype text`` or ``LABEL text``.

    Args:
        dataset_path: Path to the dataset file.
        classes: 52 for fine-grained COSC labels; 6 to collapse to main categories.

    Returns:
        Tuple of (texts, labels).
    """
    x: list[str] = []
    y: list[str] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            label, text = parts
            if classes == 6:
                label = label.split(":")[0]
            x.append(text)
            y.append(label)
    return x, y
