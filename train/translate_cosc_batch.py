#!/usr/bin/env python3
"""Batch translate COSC dataset (5 sentences per request) - much faster."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

MODEL_REPO = "DZgas/Tower-Plus-2B-GGUF"
QUANT_FILES = {
    "Q3_K_M": "Tower-Plus-2B.Q3_K_M.gguf",
    "Q4_K_M": "Tower-Plus-2B.Q4_K_M.gguf",
    "Q5_K_M": "Tower-Plus-2B.Q5_K_M.gguf",
    "Q6_K": "Tower-Plus-2B.Q6_K.gguf",
    "Q8_0": "Tower-Plus-2B.Q8_0.gguf",
}
DEFAULT_QUANT = "Q4_K_M"

LANGUAGES = {
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese (Portugal)",
    "nl": "Dutch",
}

BASE_DIR = Path("/home/miro/PycharmProjects/DEPRECATED/little_questions/train/clean_data")
EN_FILE = BASE_DIR / "raw_questions_EN_balanced_0.8.0.txt"

BATCH_SIZE = 5

PROMPT_TMPL = (
    "Translate the following English texts to {target_lang} (one per line):\n"
    "{texts}\n"
    "{target_lang} translations:\n"
)

def load_model(quant, n_gpu_layers, verbose=False):
    try:
        from llama_cpp import Llama
    except ImportError:
        print("ERROR: llama-cpp-python is not installed.", file=sys.stderr)
        sys.exit(1)
    
    filename = QUANT_FILES.get(quant)
    if filename is None:
        print(f"ERROR: unknown quantization '{quant}'", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading model {MODEL_REPO} / {filename} (gpu_layers={n_gpu_layers}) …")
    model = Llama.from_pretrained(
        repo_id=MODEL_REPO,
        filename=filename,
        n_ctx=512,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose,
    )
    print("Model loaded.\n")
    return model

def translate_batch(model, texts, target_lang):
    """Translate a batch of up to 5 sentences at once."""
    batch_str = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    user_msg = PROMPT_TMPL.format(target_lang=target_lang, texts=batch_str)
    
    try:
        result = model.create_chat_completion(
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=512,
            temperature=0.0,
            top_p=0.95,
        )
        response = result["choices"][0]["message"]["content"].strip()
        
        # Parse numbered responses
        translations = []
        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Try to extract text after number
            if line and line[0].isdigit():
                # Format: "1. translation text"
                parts = line.split(".", 1)
                if len(parts) == 2:
                    translations.append(parts[1].strip())
            else:
                # Fallback: treat whole line as translation
                translations.append(line)
        
        # If we got fewer translations than expected, pad with originals
        while len(translations) < len(texts):
            translations.append(texts[len(translations)])
        
        return translations[:len(texts)]
    except Exception as exc:
        print(f"    [warn] batch translation failed ({exc}); keeping originals", file=sys.stderr)
        return texts

def load_en_entries():
    """Load (label, text) pairs from raw_questions_EN_balanced_0.8.0.txt"""
    entries = []
    with open(EN_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ', 1)
            if len(parts) == 2:
                label, text_quoted = parts
                text = text_quoted.strip('"\'')
                entries.append((label, text))
    return entries

def write_outputs(lang_code, entries):
    """Write translated CSV and TXT files"""
    csv_path = BASE_DIR / f"raw_questions_{lang_code.upper()}_balanced_0.8.0.csv"
    txt_path = BASE_DIR / f"raw_questions_{lang_code.upper()}_balanced_0.8.0.txt"
    
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "text"])
        writer.writeheader()
        for label, _en, translated in entries:
            writer.writerow({"label": label, "text": translated})
    
    with open(txt_path, "w", encoding="utf-8") as f:
        for label, _en, translated in entries:
            f.write(f'{label} """{translated}"""\n')
    
    print(f"  Wrote {csv_path.name} and {txt_path.name}")

def ckpt_path(lang_code):
    return BASE_DIR / f".ckpt_cosc_{lang_code}.json"

def load_checkpoint(lang_code):
    p = ckpt_path(lang_code)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)

def save_checkpoint(lang_code, translations):
    with open(ckpt_path(lang_code), "w", encoding="utf-8") as f:
        json.dump(translations, f, ensure_ascii=False)

def delete_checkpoint(lang_code):
    p = ckpt_path(lang_code)
    if p.exists():
        p.unlink()

def translate_language(model, lang_code, target_lang_name, en_entries, resume, checkpoint_every=500):
    total = len(en_entries)
    print(f"\n{'='*60}")
    print(f"Translating EN → {target_lang_name} ({lang_code.upper()})  [{total} sentences]")
    print(f"Batch size: {BATCH_SIZE} sentences per request")
    print(f"{'='*60}")
    
    translations = None
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
    batches_done = 0
    
    i = 0
    while i < total:
        # Skip already translated
        if translations[i] is not None:
            i += 1
            continue
        
        # Gather batch
        batch_indices = []
        batch_texts = []
        j = i
        while j < total and len(batch_texts) < BATCH_SIZE:
            if translations[j] is None:
                batch_texts.append(en_entries[j][1])
                batch_indices.append(j)
            j += 1
        
        if not batch_texts:
            i = j
            continue
        
        # Translate batch
        translated = translate_batch(model, batch_texts, target_lang_name)
        for idx, trans in zip(batch_indices, translated):
            translations[idx] = trans
        
        since_ckpt += len(batch_texts)
        batches_done += 1
        
        # Progress report
        done = sum(1 for t in translations if t is not None)
        if done % 100 == 0 or done == total:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else float("inf")
            eta_str = f"{eta/60:.0f}m" if eta < 3600 else f"{eta/3600:.1f}h"
            print(f"  {done:>5}/{total}  [{done/total*100:.1f}%]  {rate:.1f} sent/s  ETA {eta_str}")
        
        if since_ckpt >= checkpoint_every:
            save_checkpoint(lang_code, translations)
            since_ckpt = 0
        
        i = j
    
    combined = [
        (label, text, trans)
        for (label, text), trans in zip(en_entries, translations)
        if trans is not None
    ]
    write_outputs(lang_code, combined)
    delete_checkpoint(lang_code)
    
    elapsed = time.time() - t0
    print(f"  Done in {elapsed/60:.1f} min ({batches_done} batches)")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch translate COSC dataset using Tower-Plus-2B-GGUF."
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        choices=list(LANGUAGES),
        default=list(LANGUAGES),
        help="Language codes to translate",
    )
    parser.add_argument(
        "--quant",
        choices=list(QUANT_FILES),
        default=DEFAULT_QUANT,
        help=f"GGUF quantization (default: {DEFAULT_QUANT})",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Disable GPU offload (CPU only)",
    )
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=-1,
        help="GPU layers to offload (-1 = all, 0 = CPU only)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore checkpoints and retranslate from scratch",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable llama-cpp verbose output",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not EN_FILE.exists():
        print(f"ERROR: source file not found: {EN_FILE}", file=sys.stderr)
        sys.exit(1)
    
    en_entries = load_en_entries()
    print(f"Loaded {len(en_entries)} entries from {EN_FILE.name}")
    
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
            checkpoint_every=500,
        )
    
    print("\nAll done.")

if __name__ == "__main__":
    main()
