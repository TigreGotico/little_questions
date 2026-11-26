"""Yes/no answer dataset loader for little_questions training scripts.

Source: HuggingFace dataset ``TigreGotico/yes-no-multilingual``
Labels: ``yes``, ``no``, ``maybe`` (the ``None`` agreement entries become ``maybe``)
"""

from __future__ import annotations

HF_DATASET_ID = "TigreGotico/yes-no-multilingual"
LABELS = ["yes", "no", "maybe"]


def load_yesno_hf(
    lang: str | None = None,
    split: str = "train",
) -> tuple[list[str], list[str]]:
    """Load yes/no dataset from HuggingFace.

    Args:
        lang: ISO 639-1 language code to filter (e.g. ``"en"``).
              ``None`` loads all languages (multilingual).
        split: HuggingFace split name (default ``"train"``).

    Returns:
        Tuple of (utterances, labels) where labels are ``yes``/``no``/``maybe``.
    """
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET_ID, split=split)
    x: list[str] = []
    y: list[str] = []
    for row in ds:
        if lang and row["language"] != lang:
            continue
        utterance = (row["utterance"] or "").strip()
        if not utterance:
            continue
        label = row["agreement"] if row["agreement"] else "maybe"
        x.append(utterance)
        y.append(label)
    return x, y
