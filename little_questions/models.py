"""Model path resolution for little_questions.

Resolution order for every model lookup:
  1. Bundled models shipped with the package (``little_questions/models/``)
  2. User cache (``~/.local/share/little_questions/``)
  3. HuggingFace Hub download (requires ``huggingface_hub`` and internet access)

HuggingFace is therefore fully optional — the default EN models ship bundled
so the package works offline out of the box.  Users can override with custom
models by placing them in the cache directory before first use.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

HF_REPO_ID = "TigreGotico/eat-classifiers"
HF_SENTENCE_TYPE_REPO_ID = "TigreGotico/sentence-types"
HF_YESNO_REPO_ID = "TigreGotico/yes-no-classifiers"

# Bundled models live alongside this file
_BUNDLED_DIR = Path(__file__).parent / "models"

# User override / download cache
_CACHE_DIR = Path(os.path.expanduser("~/.local/share/little_questions"))

SENTENCE_TYPE_ONNX: dict[str, str] = {
    "en": "sentence_type_EN_0.8.0.onnx",
    "de": "sentence_type_DE_0.8.0.onnx",
    "es": "sentence_type_ES_0.8.0.onnx",
    "fr": "sentence_type_FR_0.8.0.onnx",
    "it": "sentence_type_IT_0.8.0.onnx",
    "nl": "sentence_type_NL_0.8.0.onnx",
    "pt": "sentence_type_PT_0.8.0.onnx",
}


def _copy_flat(src: Path, dst: Path) -> None:
    if src != dst and src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))


def _resolve(subdir: str, filename: str, hf_repo: str, hf_path: str) -> str | None:
    """Return a path to *filename*, checking bundled → cache → HF in order."""
    # 1. Bundled with the package
    bundled = _BUNDLED_DIR / subdir / filename
    if bundled.exists():
        return str(bundled)

    # 2. User cache (may have been placed there by a previous download or manually)
    cached = _CACHE_DIR / subdir / filename
    if cached.exists():
        return str(cached)

    # 3. HuggingFace download
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=hf_repo,
            filename=hf_path,
            local_dir=str(_CACHE_DIR / subdir),
            local_dir_use_symlinks=False,
        )
        _copy_flat(Path(downloaded), cached)
        return str(cached) if cached.exists() else downloaded
    except Exception:
        return None


def get_eat_model_path(filename: str) -> str | None:
    """Return path to an EAT ONNX model (bundled → cache → HF download)."""
    return _resolve("eat", filename, HF_REPO_ID, f"models/eat/{filename}")


def get_sentence_type_model_path(lang: str) -> str | None:
    """Return path to a sentence-type ONNX model (bundled → cache → HF download)."""
    filename = SENTENCE_TYPE_ONNX.get(lang.lower())
    if not filename:
        return None
    return _resolve("sentence_type", filename, HF_SENTENCE_TYPE_REPO_ID, filename)


def get_yesno_model_path(lang: str, version: str = "0.9.0") -> str | None:
    """Return path to a yes/no ONNX model (bundled → cache → HF download).

    Tries language-specific first, then the multilingual model.
    """
    candidates = [
        f"yesno_svm_cal_{lang.upper()}_{version}.onnx",
        f"yesno_svm_cal_multilingual_{version}.onnx",
    ]
    for filename in candidates:
        path = _resolve("yesno", filename, HF_YESNO_REPO_ID, f"models/yesno/{filename}")
        if path:
            return path
    return None
