"""Model download and caching for little_questions.

Models are downloaded from GitHub releases on first use and cached locally.
"""

from __future__ import annotations

import hashlib
import logging
import time
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

# SHA-256 digests for all v0.8.0 model files.
# Populated after models are uploaded to the release; None means unverified.
MODEL2SHA256: dict[str, Optional[str]] = {lang: None for lang in LANG2MODEL}

_DOWNLOAD_CHUNK_SIZE = 8192
_DOWNLOAD_MAX_RETRIES = 3
_DOWNLOAD_RETRY_DELAY = 2.0  # seconds


def get_cache_dir() -> str:
    """Return the XDG data directory used to cache downloaded models."""
    return XDG.save_data_path("little_questions")


def _sha256(path: str) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_DOWNLOAD_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_checksum(path: str, model_id: str) -> None:
    """Raise ``ValueError`` if the file's SHA-256 does not match the manifest.

    A missing manifest entry (``None``) is logged as a warning and skipped.
    """
    expected = MODEL2SHA256.get(model_id)
    if expected is None:
        LOG.warning(
            "No checksum registered for model %r — skipping verification. "
            "Update MODEL2SHA256 in little_questions/models/__init__.py after "
            "uploading release assets.",
            model_id,
        )
        return
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"Checksum mismatch for {path!r}: "
            f"expected {expected}, got {actual}. "
            "The file may be corrupted or tampered with. Delete it and re-download."
        )


def _download_file(url: str, dest: str) -> None:
    """Download *url* to *dest* using chunked streaming with retry.

    Retries up to ``_DOWNLOAD_MAX_RETRIES`` times on transient errors.
    """
    for attempt in range(1, _DOWNLOAD_MAX_RETRIES + 1):
        try:
            LOG.info("Downloading %s (attempt %d/%d)", url, attempt, _DOWNLOAD_MAX_RETRIES)
            with requests.get(url, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                downloaded = 0
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded * 100 // total
                            LOG.debug("  %d%%", pct)
            LOG.info("Saved %s (%d bytes)", dest, downloaded)
            return
        except requests.RequestException as exc:
            LOG.warning("Download attempt %d failed: %s", attempt, exc)
            if attempt < _DOWNLOAD_MAX_RETRIES:
                time.sleep(_DOWNLOAD_RETRY_DELAY)
    raise RuntimeError(
        f"Failed to download {url} after {_DOWNLOAD_MAX_RETRIES} attempts."
    )


def download(model_id: str, force: bool = False) -> str:
    """Download a model file from GitHub releases and verify its checksum.

    Args:
        model_id: Language code (e.g. ``"en"``) or bare filename.
        force: Re-download even if the file already exists.

    Returns:
        Absolute path to the downloaded file.
    """
    filename = LANG2MODEL.get(model_id, model_id)
    if not filename.endswith(".onnx"):
        filename = f"{filename}.onnx"

    url = MODEL2URL.get(model_id) or (
        f"https://github.com/OpenJarbas/little_questions/releases/download/0.8.0/{filename}"
    )
    path = join(get_cache_dir(), filename)

    if isfile(path) and not force:
        LOG.info("Model already cached: %s", filename)
        _verify_checksum(path, model_id)
        return path

    _download_file(url, path)
    _verify_checksum(path, model_id)
    return path


def get_model_path(model: str = "en") -> str:
    """Return the local path to a model, downloading it if necessary.

    Args:
        model: Language code (``"en"``, ``"es"``, …), bare filename, or
               absolute path to an existing file.

    Returns:
        Absolute path to the model file.

    Raises:
        ValueError: If *model* is not a recognised language code.
    """
    if isfile(model):
        return model

    if model in LANG2MODEL:
        filename = LANG2MODEL[model]
        path = join(get_cache_dir(), filename)
        if isfile(path):
            return path
        return download(model)

    raise ValueError(
        f"Unknown model identifier: {model!r}. "
        f"Expected one of: {', '.join(LANG2MODEL)}"
    )


def download_en() -> None:
    """Download English models and required NLTK data."""
    download("en")
    download("en_small")
    nltk.download("maxent_ne_chunker", quiet=True)
    nltk.download("words", quiet=True)


def download_pt() -> None:
    """Download Portuguese models."""
    download("pt")
    download("pt_small")


def download_es() -> None:
    """Download Spanish models."""
    download("es")
    download("es_small")


def download_fr() -> None:
    """Download French models."""
    download("fr")
    download("fr_small")


def download_it() -> None:
    """Download Italian models."""
    download("it")
    download("it_small")


def download_de() -> None:
    """Download German models."""
    download("de")
    download("de_small")


def download_ca() -> None:
    """Download Catalan models."""
    download("ca")
    download("ca_small")


def download_nl() -> None:
    """Download Dutch models."""
    download("nl")
    download("nl_small")
