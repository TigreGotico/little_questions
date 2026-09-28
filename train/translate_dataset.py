#!/usr/bin/env python3
"""Translate sentence_types_EN.csv to multiple languages using Tower-Plus-2B-GGUF.

Tower-Plus-2B is a translation-specialized LLM from Unbabel, quantized to GGUF
format by DZgas. It is loaded via llama-cpp-python and run locally — no API key
or internet connection required after the one-time model download.

Usage:
    # Translate all languages (resumes from checkpoint automatically):
    uv run python train/translate_dataset.py

    # Translate specific languages only:
    uv run python train/translate_dataset.py --langs es fr de

    # Use a different quantization (default: Q4_K_M):
    uv run python train/translate_dataset.py --quant Q5_K_M

    # Force CPU only (no GPU offload):
    uv run python train/translate_dataset.py --cpu

    # Wipe checkpoints and retranslate from scratch:
    uv run python train/translate_dataset.py --no-resume

Install dependencies:
    uv pip install llama-cpp-python

    # For GPU acceleration (optional, requires CUDA):
    CMAKE_ARGS="-DGGML_CUDA=on" uv pip install --force-reinstall llama-cpp-python

Notes:
    - Catalan (CA) is not in Tower-Plus's officially supported language list;
      translations will be attempted but quality may be lower than other languages.
    - Checkpoint files (*.ckpt.json) are written to train/clean_data/ and removed
      automatically when a language is fully translated.
    - Each language takes roughly 2–5 hours on CPU, 20–40 minutes with GPU offload.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_REPO = "DZgas/Tower-Plus-2B-GGUF"

#: Map quantization shorthand → filename pattern in the GGUF repo.
QUANT_FILES: Dict[str, str] = {
    "Q3_K_M": "Tower-Plus-2B.Q3_K_M.gguf",
    "Q4_K_M": "Tower-Plus-2B.Q4_K_M.gguf",
    "Q5_K_M": "Tower-Plus-2B.Q5_K_M.gguf",
    "Q6_K":   "Tower-Plus-2B.Q6_K.gguf",
    "Q8_0":   "Tower-Plus-2B.Q8_0.gguf",
}
DEFAULT_QUANT = "Q4_K_M"

#: Target languages.  Key = ISO code used in filenames; value = full name for
#: the Tower prompt.  Catalan is marked with a warning at runtime.
LANGUAGES: Dict[str, str] = {
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese (Portugal)",
    "nl": "Dutch",
    "ca": "Catalan",  # not in Tower's official list; quality may vary
}

#: Languages not in Tower-Plus's official training set.
UNOFFICIAL_LANGS = {"ca"}

BASE_DIR = Path(__file__).parent / "clean_data"
EN_CSV   = BASE_DIR / "sentence_types_EN.csv"

#: Tower translation prompt template (no system message needed).
PROMPT_TMPL = (
    "Translate the following English source text to {target_lang}:\n"
    "English: {text}\n"
    "{target_lang}: "
)

# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_model(quant: str, n_gpu_layers: int, verbose: bool = False):
    """Download (if needed) and load the GGUF model.

    Args:
        quant: Quantization shorthand, e.g. ``"Q4_K_M"``.
        n_gpu_layers: Number of layers to offload to GPU.  ``-1`` = all.
        verbose: Pass through to llama-cpp for debug output.

    Returns:
        A ``llama_cpp.Llama`` instance.

    Raises:
        SystemExit: If llama-cpp-python is not installed.
    """
    try:
        from llama_cpp import Llama  # type: ignore
    except ImportError:
        print(
            "ERROR: llama-cpp-python is not installed.\n"
            "Install with:  uv pip install llama-cpp-python\n"
            "GPU (CUDA):    CMAKE_ARGS='-DGGML_CUDA=on' "
            "uv pip install --force-reinstall llama-cpp-python",
            file=sys.stderr,
        )
        sys.exit(1)

    filename = QUANT_FILES.get(quant)
    if filename is None:
        print(
            f"ERROR: unknown quantization '{quant}'. "
            f"Choose from: {', '.join(QUANT_FILES)}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading model  {MODEL_REPO} / {filename}  (gpu_layers={n_gpu_layers}) …")
    model = Llama.from_pretrained(
        repo_id=MODEL_REPO,
        filename=filename,
        n_ctx=512,        # single sentence fits easily; small ctx = faster
        n_gpu_layers=n_gpu_layers,
        verbose=verbose,
    )
    print("Model loaded.\n")
    return model


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def translate_one(model, text: str, target_lang: str) -> str:
    """Translate a single sentence using Tower-Plus-2B.

    Uses the Gemma2 chat completion API so that proper ``<start_of_turn>``
    tokens are applied, which Tower-Plus-2B requires to follow instructions.

    Args:
        model: Loaded ``llama_cpp.Llama`` instance.
        text: English source sentence.
        target_lang: Full language name used in the Tower prompt, e.g. ``"French"``.

    Returns:
        Translated string, or the original *text* on failure.
    """
    user_msg = PROMPT_TMPL.format(target_lang=target_lang, text=text)
    try:
        result = model.create_chat_completion(
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=256,
            temperature=0.0,
            top_p=0.95,
        )
        translation: str = result["choices"][0]["message"]["content"].strip()
        # Strip any repeated language label the model occasionally prefixes.
        prefix = f"{target_lang}:"
        if translation.lower().startswith(prefix.lower()):
            translation = translation[len(prefix):].strip()
        return translation if translation else text
    except Exception as exc:  # noqa: BLE001
        print(f"    [warn] translation failed ({exc}); keeping original", file=sys.stderr)
        return text


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def load_en_entries() -> List[Tuple[str, str]]:
    """Load ``(label, text)`` pairs from the English CSV.

    Returns:
        List of ``(label, text)`` tuples in file order.
    """
    with open(EN_CSV, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [(row["label"], row["text"]) for row in reader]


def write_outputs(lang_code: str, entries: List[Tuple[str, str, str]]) -> None:
    """Write translated CSV and TXT files.

    Args:
        lang_code: ISO language code, e.g. ``"es"``.
        entries: List of ``(label, en_text, translated_text)`` tuples.
    """
    csv_path = BASE_DIR / f"sentence_types_{lang_code.upper()}.csv"
    txt_path = BASE_DIR / f"sentence_types_{lang_code.upper()}.txt"

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "text"])
        writer.writeheader()
        for label, _en, translated in entries:
            writer.writerow({"label": label, "text": translated})

    with open(txt_path, "w", encoding="utf-8") as f:
        for i, (label, _en, translated) in enumerate(entries, 1):
            f.write(f"{i}: {label} {translated}\n")

    print(f"  Wrote {csv_path.name} and {txt_path.name}")


# ---------------------------------------------------------------------------
# Checkpoint helpers  (allow resuming interrupted runs)
# ---------------------------------------------------------------------------

def ckpt_path(lang_code: str) -> Path:
    """Return the checkpoint file path for *lang_code*."""
    return BASE_DIR / f".ckpt_{lang_code}.json"


def load_checkpoint(lang_code: str) -> Optional[List[Optional[str]]]:
    """Load existing checkpoint for *lang_code*, or return ``None``.

    Returns:
        A list of translated strings (``None`` where not yet translated), or
        ``None`` if no checkpoint exists.
    """
    p = ckpt_path(lang_code)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(lang_code: str, translations: List[Optional[str]]) -> None:
    """Persist the current translation progress for *lang_code*."""
    with open(ckpt_path(lang_code), "w", encoding="utf-8") as f:
        json.dump(translations, f, ensure_ascii=False)


def delete_checkpoint(lang_code: str) -> None:
    """Remove the checkpoint file once translation is complete."""
    p = ckpt_path(lang_code)
    if p.exists():
        p.unlink()


# ---------------------------------------------------------------------------
# Per-language translation loop
# ---------------------------------------------------------------------------

def translate_language(
    model,
    lang_code: str,
    target_lang_name: str,
    en_entries: List[Tuple[str, str]],
    resume: bool,
    checkpoint_every: int = 100,
) -> None:
    """Translate all entries to one language and write output files.

    Args:
        model: Loaded ``llama_cpp.Llama`` instance.
        lang_code: ISO code, e.g. ``"es"``.
        target_lang_name: Full name used in the Tower prompt, e.g. ``"Spanish"``.
        en_entries: ``(label, text)`` pairs from the English source.
        resume: If ``True``, load and continue from checkpoint.
        checkpoint_every: Save progress after this many new translations.
    """
    total = len(en_entries)
    print(f"\n{'='*60}")
    print(f"Translating EN → {target_lang_name} ({lang_code.upper()})  [{total} sentences]")
    if lang_code in UNOFFICIAL_LANGS:
        print(
            f"  ⚠  {target_lang_name} is not in Tower-Plus's official language list; "
            "quality may be lower."
        )
    print(f"{'='*60}")

    # Load or initialise translation list
    translations: List[Optional[str]]
    if resume:
        saved = load_checkpoint(lang_code)
        if saved is not None and len(saved) == total:
            done_count = sum(1 for t in saved if t is not None)
            print(f"  Resuming from checkpoint ({done_count}/{total} already done)")
            translations = saved
        else:
            if saved is not None:
                print("  Checkpoint length mismatch — starting fresh")
            translations = [None] * total
    else:
        translations = [None] * total

    t0 = time.time()
    since_ckpt = 0

    for i, (label, text) in enumerate(en_entries):
        if translations[i] is not None:
            continue  # already translated in a previous run

        translations[i] = translate_one(model, text, target_lang_name)
        since_ckpt += 1

        # Progress report
        done = sum(1 for t in translations if t is not None)
        if done % 50 == 0 or done == total:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else float("inf")
            eta_str = f"{eta/60:.0f}m" if eta < 3600 else f"{eta/3600:.1f}h"
            print(
                f"  {done:>5}/{total}  "
                f"[{done/total*100:.1f}%]  "
                f"{rate:.1f} sent/s  ETA {eta_str}"
            )

        # Periodic checkpoint
        if since_ckpt >= checkpoint_every:
            save_checkpoint(lang_code, translations)
            since_ckpt = 0

    # Final write
    combined = [
        (label, text, trans)
        for (label, text), trans in zip(en_entries, translations)
        if trans is not None
    ]
    write_outputs(lang_code, combined)
    delete_checkpoint(lang_code)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed/60:.1f} min")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Translate sentence_types_EN.csv to multiple languages "
                    "using Tower-Plus-2B-GGUF."
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        choices=list(LANGUAGES),
        default=list(LANGUAGES),
        metavar="LANG",
        help=f"Language codes to translate (default: all). Choices: {', '.join(LANGUAGES)}",
    )
    parser.add_argument(
        "--quant",
        choices=list(QUANT_FILES),
        default=DEFAULT_QUANT,
        help=f"GGUF quantization to use (default: {DEFAULT_QUANT}).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Disable GPU offload (force CPU-only inference).",
    )
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=-1,
        dest="gpu_layers",
        help="Number of model layers to offload to GPU (-1 = all, 0 = CPU only).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        dest="no_resume",
        help="Ignore existing checkpoints and retranslate from scratch.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        dest="checkpoint_every",
        metavar="N",
        help="Save checkpoint every N translated sentences (default: 100).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable llama-cpp verbose output.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    if not EN_CSV.exists():
        print(f"ERROR: source file not found: {EN_CSV}", file=sys.stderr)
        sys.exit(1)

    en_entries = load_en_entries()
    print(f"Loaded {len(en_entries)} entries from {EN_CSV.name}")

    n_gpu_layers = 0 if args.cpu else args.gpu_layers
    model = load_model(args.quant, n_gpu_layers, verbose=args.verbose)

    for lang_code in args.langs:
        target_lang_name = LANGUAGES[lang_code]
        translate_language(
            model=model,
            lang_code=lang_code,
            target_lang_name=target_lang_name,
            en_entries=en_entries,
            resume=not args.no_resume,
            checkpoint_every=args.checkpoint_every,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
