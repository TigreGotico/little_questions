"""EAT dataset loader for little_questions training scripts.

Primary source: HuggingFace dataset ``TigreGotico/EAT`` so results are reproducible.
Fallback: local TSV file with columns label, question, answer, lang.
"""

from __future__ import annotations

import csv

HF_DATASET_ID = "TigreGotico/EAT"


def load_eat_hf(classes: int = 53, split: str = "train") -> tuple[list[str], list[str]]:
    """Load EAT from HuggingFace Hub (``TigreGotico/EAT``).

    Args:
        classes: 53 keeps fine-grained labels (e.g. ``HUM:ind``);
                 7 collapses to main category (e.g. ``HUM``).
        split: HuggingFace split name (default ``"train"``).
    """
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET_ID, split=split)
    x: list[str] = []
    y: list[str] = []
    for row in ds:
        label = row["label"].strip()
        question = row["question"].strip()
        if not label or not question:
            continue
        if classes == 7:
            label = label.split(":")[0]
        x.append(question)
        y.append(label)
    return x, y


def load_eat(dataset_path: str, classes: int = 53) -> tuple[list[str], list[str]]:
    """Load EAT from a local TSV file.

    TSV format (with header): label \\t question \\t answer \\t lang

    Args:
        dataset_path: Path to EAT.tsv.
        classes: 53 keeps fine-grained labels; 7 collapses to main category.
    """
    x: list[str] = []
    y: list[str] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            label = row["label"].strip()
            question = row["question"].strip()
            if not label or not question:
                continue
            if classes == 7:
                label = label.split(":")[0]
            x.append(question)
            y.append(label)
    return x, y
