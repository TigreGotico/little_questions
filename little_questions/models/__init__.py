"""Model download and path resolution for little_questions.

Models are downloaded from HuggingFace Hub on first use and cached in the
default HF cache (~/.cache/huggingface/hub/).  Locally-trained models in the
XDG data directory (~/.local/share/little_questions/) take priority.
"""

from __future__ import annotations

import logging
from os.path import isfile, join
from typing import Optional

from xdg import BaseDirectory as XDG

LOG = logging.getLogger(__name__)

HF_REPO_ID = "TigreGotico/little-questions"

# Filename for each model identifier (used for XDG path lookup).
LANG2MODEL: dict[str, str] = {
    # COSC 52-class
    "en":    "questions52_svm_EN_0.8.0.onnx",
    "es":    "questions52_svm_ES_0.8.0.onnx",
    "pt":    "questions52_svm_PT_0.8.0.onnx",
    "ca":    "questions52_svm_CA_0.8.0.onnx",
    "fr":    "questions52_svm_FR_0.8.0.onnx",
    "de":    "questions52_svm_DE_0.8.0.onnx",
    "it":    "questions52_svm_IT_0.8.0.onnx",
    "nl":    "questions52_svm_NL_0.8.0.onnx",
    # COSC 6-class
    "en_small": "questions6_svm_EN_0.8.0.onnx",
    "es_small": "questions6_svm_ES_0.8.0.onnx",
    "pt_small": "questions6_svm_PT_0.8.0.onnx",
    "ca_small": "questions6_svm_CA_0.8.0.onnx",
    "fr_small": "questions6_svm_FR_0.8.0.onnx",
    "de_small": "questions6_svm_DE_0.8.0.onnx",
    "it_small": "questions6_svm_IT_0.8.0.onnx",
    "nl_small": "questions6_svm_NL_0.8.0.onnx",
    # Sentence-type classifiers
    "sentence_type_en": "sentence_type_EN_0.8.0.onnx",
    "sentence_type_es": "sentence_type_ES_0.8.0.onnx",
    "sentence_type_pt": "sentence_type_PT_0.8.0.onnx",
    "sentence_type_ca": "sentence_type_CA_0.8.0.onnx",
    "sentence_type_fr": "sentence_type_FR_0.8.0.onnx",
    "sentence_type_de": "sentence_type_DE_0.8.0.onnx",
    "sentence_type_it": "sentence_type_IT_0.8.0.onnx",
    "sentence_type_nl": "sentence_type_NL_0.8.0.onnx",
}

# Path within the HF repo for each model identifier.
_HF_PATHS: dict[str, str] = {
    "en":    "models/cosc/svm/52/en/questions52_svm_EN_0.8.0.onnx",
    "es":    "models/cosc/svm/52/es/questions52_svm_ES_0.8.0.onnx",
    "pt":    "models/cosc/svm/52/pt/questions52_svm_PT_0.8.0.onnx",
    "ca":    "models/cosc/svm/52/ca/questions52_svm_CA_0.8.0.onnx",
    "fr":    "models/cosc/svm/52/fr/questions52_svm_FR_0.8.0.onnx",
    "de":    "models/cosc/svm/52/de/questions52_svm_DE_0.8.0.onnx",
    "it":    "models/cosc/svm/52/it/questions52_svm_IT_0.8.0.onnx",
    "nl":    "models/cosc/svm/52/nl/questions52_svm_NL_0.8.0.onnx",
    "en_small": "models/cosc/svm/6/en/questions6_svm_EN_0.8.0.onnx",
    "es_small": "models/cosc/svm/6/es/questions6_svm_ES_0.8.0.onnx",
    "pt_small": "models/cosc/svm/6/pt/questions6_svm_PT_0.8.0.onnx",
    "ca_small": "models/cosc/svm/6/ca/questions6_svm_CA_0.8.0.onnx",
    "fr_small": "models/cosc/svm/6/fr/questions6_svm_FR_0.8.0.onnx",
    "de_small": "models/cosc/svm/6/de/questions6_svm_DE_0.8.0.onnx",
    "it_small": "models/cosc/svm/6/it/questions6_svm_IT_0.8.0.onnx",
    "nl_small": "models/cosc/svm/6/nl/questions6_svm_NL_0.8.0.onnx",
    "sentence_type_en": "models/sentence_type/svm/en/sentence_type_EN_0.8.0.onnx",
    "sentence_type_es": "models/sentence_type/svm/es/sentence_type_ES_0.8.0.onnx",
    "sentence_type_pt": "models/sentence_type/svm/pt/sentence_type_PT_0.8.0.onnx",
    "sentence_type_ca": "models/sentence_type/svm/ca/sentence_type_CA_0.8.0.onnx",
    "sentence_type_fr": "models/sentence_type/svm/fr/sentence_type_FR_0.8.0.onnx",
    "sentence_type_de": "models/sentence_type/svm/de/sentence_type_DE_0.8.0.onnx",
    "sentence_type_it": "models/sentence_type/svm/it/sentence_type_IT_0.8.0.onnx",
    "sentence_type_nl": "models/sentence_type/svm/nl/sentence_type_NL_0.8.0.onnx",
}


def get_cache_dir() -> str:
    """Return the XDG data directory for locally-trained models."""
    return XDG.save_data_path("little_questions")


def _hf_download(model_id: str) -> str:
    """Download a model from HuggingFace Hub and return its local cache path."""
    from huggingface_hub import hf_hub_download

    hf_path = _HF_PATHS.get(model_id)
    if not hf_path:
        raise ValueError(
            f"No HuggingFace path registered for {model_id!r}. "
            f"Known IDs: {', '.join(_HF_PATHS)}"
        )
    LOG.info("Downloading %s from %s/%s", model_id, HF_REPO_ID, hf_path)
    return hf_hub_download(repo_id=HF_REPO_ID, filename=hf_path)


def get_model_path(model: str = "en") -> str:
    """Return the local path to a COSC ONNX model, downloading if necessary.

    Resolution order:
    1. If *model* is an existing filesystem path — return as-is.
    2. If the model filename exists in the XDG data directory — return it
       (locally-trained models take priority over downloaded ones).
    3. Download from HuggingFace Hub (cached in HF default cache).

    Args:
        model: Language code (``"en"``, ``"es"``, …) or absolute file path.

    Raises:
        ValueError: If *model* is not a recognised identifier.
    """
    if isfile(model):
        return model

    if model not in LANG2MODEL:
        raise ValueError(
            f"Unknown model identifier: {model!r}. "
            f"Expected one of: {', '.join(LANG2MODEL)}"
        )

    xdg_path = join(get_cache_dir(), LANG2MODEL[model])
    if isfile(xdg_path):
        LOG.debug("Using locally-trained model: %s", xdg_path)
        return xdg_path

    return _hf_download(model)


def get_sentence_type_model_path(lang: str = "en") -> Optional[str]:
    """Return the local path to the sentence-type ONNX model for *lang*.

    Returns None when the model is not available (neither locally nor on HF).
    Callers should fall back to the built-in heuristic scorer.
    """
    model_id = f"sentence_type_{lang.lower()}"
    if model_id not in LANG2MODEL:
        LOG.debug("No sentence-type model registered for lang=%s", lang)
        return None
    try:
        xdg_path = join(get_cache_dir(), LANG2MODEL[model_id])
        if isfile(xdg_path):
            return xdg_path
        return _hf_download(model_id)
    except Exception as exc:
        LOG.warning("Could not load sentence-type model for %s: %s", lang, exc)
        return None


# ---------------------------------------------------------------------------
# Convenience download helpers
# ---------------------------------------------------------------------------

def download(model_id: str, force: bool = False) -> str:
    """Download a model from HuggingFace Hub.

    If *model_id* already exists in the XDG directory and *force* is False,
    the XDG path is returned without downloading.

    Args:
        model_id: A key from ``LANG2MODEL``.
        force: Re-download even if a local copy exists.

    Returns:
        Absolute path to the model file (HF cache or XDG).
    """
    if model_id not in LANG2MODEL:
        raise ValueError(f"Unknown model_id: {model_id!r}")

    if not force:
        xdg_path = join(get_cache_dir(), LANG2MODEL[model_id])
        if isfile(xdg_path):
            return xdg_path

    return _hf_download(model_id)


def download_en() -> None:
    """Download English COSC and sentence-type models."""
    download("en")
    download("en_small")
    download("sentence_type_en")


def download_es() -> None:
    """Download Spanish COSC and sentence-type models."""
    download("es")
    download("es_small")
    download("sentence_type_es")


def download_pt() -> None:
    """Download Portuguese COSC and sentence-type models."""
    download("pt")
    download("pt_small")
    download("sentence_type_pt")


def download_ca() -> None:
    """Download Catalan COSC and sentence-type models."""
    download("ca")
    download("ca_small")
    download("sentence_type_ca")


def download_fr() -> None:
    """Download French COSC and sentence-type models."""
    download("fr")
    download("fr_small")
    download("sentence_type_fr")


def download_de() -> None:
    """Download German COSC and sentence-type models."""
    download("de")
    download("de_small")
    download("sentence_type_de")


def download_it() -> None:
    """Download Italian COSC and sentence-type models."""
    download("it")
    download("it_small")
    download("sentence_type_it")


def download_nl() -> None:
    """Download Dutch COSC and sentence-type models."""
    download("nl")
    download("nl_small")
    download("sentence_type_nl")
