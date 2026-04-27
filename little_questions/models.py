"""Model download and path resolution for little_questions."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

HF_REPO_ID = "TigreGotico/eat-classifiers"
HF_SENTENCE_TYPE_REPO_ID = "TigreGotico/sentence-types"
HF_YESNO_REPO_ID = "TigreGotico/yes-no-classifiers"

_CACHE_DIR = Path(os.path.expanduser("~/.local/share/little_questions"))

# Sentence-type ONNX filenames in HF_SENTENCE_TYPE_REPO_ID
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


def get_eat_model_path(filename: str) -> str | None:
    """Return local path to an EAT ONNX model, downloading from HF if needed."""
    local = _CACHE_DIR / "eat" / filename
    if local.exists():
        return str(local)

    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=f"models/eat/{filename}",
            local_dir=str(_CACHE_DIR / "eat"),
            local_dir_use_symlinks=False,
        )
        _copy_flat(Path(downloaded), local)
        return str(local) if local.exists() else downloaded
    except Exception:
        return None


def get_yesno_model_path(lang: str, version: str = "0.9.0") -> str | None:
    """Return local path to a yes/no ONNX model, downloading from HF if needed.

    Tries language-specific model first (``yesno_svm_cal_{LANG}_{version}.onnx``),
    then falls back to the multilingual model (``yesno_svm_cal_multilingual_{version}.onnx``).
    """
    candidates = [
        f"yesno_svm_cal_{lang.upper()}_{version}.onnx",
        f"yesno_svm_cal_multilingual_{version}.onnx",
    ]
    for filename in candidates:
        local = _CACHE_DIR / "yesno" / filename
        if local.exists():
            return str(local)
        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id=HF_YESNO_REPO_ID,
                filename=f"models/yesno/{filename}",
                local_dir=str(_CACHE_DIR / "yesno"),
                local_dir_use_symlinks=False,
            )
            _copy_flat(Path(downloaded), local)
            if local.exists():
                return str(local)
        except Exception:
            continue
    return None


def get_sentence_type_model_path(lang: str) -> str | None:
    """Return local path to a sentence-type ONNX model, downloading from HF if needed."""
    filename = SENTENCE_TYPE_ONNX.get(lang.lower())
    if not filename:
        return None

    local = _CACHE_DIR / "sentence_type" / filename
    if local.exists():
        return str(local)

    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=HF_SENTENCE_TYPE_REPO_ID,
            filename=filename,
            local_dir=str(_CACHE_DIR / "sentence_type"),
            local_dir_use_symlinks=False,
        )
        _copy_flat(Path(downloaded), local)
        return str(local) if local.exists() else downloaded
    except Exception:
        return None
