"""Model download and caching for little_questions.

Models are downloaded from GitHub releases on first use and cached locally.
"""

from __future__ import annotations

import logging
from os.path import join, isfile
from typing import Optional

import nltk
import requests
from xdg import BaseDirectory as XDG

LOG = logging.getLogger(__name__)

LANG2MODEL = {
    "en": "questions52_svm_EN_0.8.0.onnx",
    "en_small": "questions6_svm_EN_0.8.0.onnx",
    "es": "questions52_svm_ES_0.8.0.onnx",
    "es_small": "questions6_svm_ES_0.8.0.onnx",
    "ca": "questions52_svm_CA_0.8.0.onnx",
    "ca_small": "questions6_svm_CA_0.8.0.onnx",
    "pt": "questions52_svm_PT_0.8.0.onnx",
    "pt_small": "questions6_svm_PT_0.8.0.onnx",
    "fr": "questions52_svm_FR_0.8.0.onnx",
    "fr_small": "questions6_svm_FR_0.8.0.onnx",
    "de": "questions52_svm_DE_0.8.0.onnx",
    "de_small": "questions6_svm_DE_0.8.0.onnx",
    "it": "questions52_svm_IT_0.8.0.onnx",
    "it_small": "questions6_svm_IT_0.8.0.onnx",
    "nl": "questions52_svm_NL_0.8.0.onnx",
    "nl_small": "questions6_svm_NL_0.8.0.onnx",
}

MODEL2URL = {
    lang: f"https://github.com/OpenJarbas/little_questions/releases/download/0.8.0/{filename}"
    for lang, filename in LANG2MODEL.items()
}


def get_cache_dir() -> str:
    """Get the cache directory for models."""
    return XDG.save_data_path("little_questions")


def download(model_id: str, force: bool = False) -> str:
    """Download a model file from GitHub releases.

    Args:
        model_id: Model identifier (e.g. "en", "questions52_EN")
        force: Force re-download even if file exists

    Returns:
        Path to downloaded file
    """
    filename = LANG2MODEL.get(model_id, model_id)
    if not filename.endswith(".onnx"):
        filename = f"{filename}.onnx"

    if model_id in MODEL2URL:
        url = MODEL2URL[model_id]
    else:
        url = f"https://github.com/OpenJarbas/little_questions/releases/download/0.8.0/{filename}"

    path = join(get_cache_dir(), filename)

    if isfile(path) and not force:
        LOG.info("Model already cached: %s", filename)
        return path

    LOG.info("Downloading model: %s", filename)
    LOG.info("URL: %s", url)

    response = requests.get(url, timeout=300)
    response.raise_for_status()

    with open(path, "wb") as f:
        f.write(response.content)

    LOG.info("Downloaded: %s (%d bytes)", path, len(response.content))
    return path


def get_model_path(model: str = "en") -> str:
    """Get path to a downloaded model, downloading if necessary.

    Args:
        model: Language code (e.g. "en", "es") or model identifier

    Returns:
        Absolute path to the model file

    Raises:
        ValueError: If model is not supported
    """
    if model.startswith("http"):
        raise NotImplementedError("downloading models from URL not supported")

    if isfile(model):
        return model

    if model in LANG2MODEL:
        filename = LANG2MODEL[model]
        path = join(get_cache_dir(), filename)
        if isfile(path):
            return path
        return download(model)

    raise ValueError(f"unknown model: {model}")


def download_en():
    """Download English models."""
    download("en")
    download("en_small")
    nltk.download("maxent_ne_chunker", quiet=True)
    nltk.download("words", quiet=True)


def download_pt():
    """Download Portuguese models."""
    download("pt")
    download("pt_small")


def download_es():
    """Download Spanish models."""
    download("es")
    download("es_small")


def download_fr():
    """Download French models."""
    download("fr")
    download("fr_small")


def download_it():
    """Download Italian models."""
    download("it")
    download("it_small")


def download_de():
    """Download German models."""
    download("de")
    download("de_small")


def download_ca():
    """Download Catalan models."""
    download("ca")
    download("ca_small")
