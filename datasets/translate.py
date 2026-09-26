#!/usr/bin/env python3
"""Batch translate COSC dataset (5 sentences per request) - much faster."""
import os
import sys

from tqdm import tqdm

MODEL_REPO = "DZgas/Tower-Plus-2B-GGUF"
QUANT_FILES = {
    "Q3_K_M": "Tower-Plus-2B.Q3_K_M.gguf",
    "Q4_K_M": "Tower-Plus-2B.Q4_K_M.gguf",
    "Q5_K_M": "Tower-Plus-2B.Q5_K_M.gguf",
    "Q6_K": "Tower-Plus-2B.Q6_K.gguf",
    "Q8_0": "Tower-Plus-2B.Q8_0.gguf",
}
DEFAULT_QUANT = "Q4_K_M"

# target langs
LANGUAGES = {
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
}

BASE_DIR = "/home/miro/AgentWorkspaces/LILACS/little_questions/datasets"
EN_FILE = f"{BASE_DIR}/question_types_EN_fixed.csv"

BATCH_SIZE = 10

PROMPT_TMPL = (
    "Translate the following English texts to {target_lang} (one per line):\n"
    "{texts}\n"
    "{target_lang} translations:\n"
)


def load_model(quant="Q4_K_M", n_gpu_layers=-1, verbose=False):
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
        n_ctx=8192,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose,
    )
    print("Model loaded.\n")
    return model


def translate_batch(model, texts, target_lang):
    """Translate a batch of up to 5 sentences at once."""
    batch_str = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    user_msg = PROMPT_TMPL.format(target_lang=target_lang, texts=batch_str)

    try:
        result = model.create_chat_completion(
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=8192,
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
            label, question, lang = line.split('\t', 2)
            entries.append((label, question))
    return entries


# ... (Previous imports and constants remain the same)

def translate_language(lang_code, target_lang_name, en_entries):
    """Translates EN entries to target language with resume support."""
    total = len(en_entries)
    out_file = f"{BASE_DIR}/question_types_{lang_code.upper()}.csv"

    # 1. Resume Logic: Load existing translations
    translated_questions = set()
    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    # We use the original English question as the key to check progress
                    # This assumes the English file is the source of truth
                    translated_questions.add(parts[1])

    # Filter out what is already done
    to_translate = [e for e in en_entries if e[1] not in translated_questions]

    if not to_translate:
        print(f"All {total} entries already translated for {lang_code.upper()}. Skipping.")
        return

    print(f"\n{'=' * 60}")
    print(f"Translating EN → {target_lang_name} ({lang_code.upper()})")
    print(f"Total: {total} | Already done: {len(translated_questions)} | Remaining: {len(to_translate)}")
    print(f"{'=' * 60}")

    model = load_model()

    # 2. Open file in append mode
    with open(out_file, "a", encoding="utf-8", buffering=1) as f:
        # If file is empty, write header
        if os.path.getsize(out_file) == 0:
            f.write("label\tquestion\tlang\n")

        # 3. Batch processing
        for i in tqdm(range(0, len(to_translate), BATCH_SIZE)):
            batch = to_translate[i: i + BATCH_SIZE]
            labels = [b[0] for b in batch]
            texts = [b[1] for b in batch]

            translations = translate_batch(model, texts, target_lang_name)

            for label, trans_text in zip(labels, translations):
                # Format: label \t translated_question \t lang_code
                f.write(f"{label}\t{trans_text}\t{lang_code}\n")


def main():
    en_entries = load_en_entries()
    if not en_entries:
        print("No English entries found to translate.")
        return

    # Skip header if it exists in the EN_FILE
    if en_entries[0][0] == "label":
        en_entries = en_entries[1:]

    for code, name in LANGUAGES.items():
        translate_language(code, name, en_entries)


if __name__ == "__main__":
    main()
