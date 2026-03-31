#!/usr/bin/env python3
"""Translate sentence types dataset while preserving labels."""

import re
from pathlib import Path

LANG_PAIRS = {
    "es": "es",
    "pt": "pt",
    "ca": "ca",
    "fr": "fr",
    "de": "de",
    "it": "it",
    "nl": "nl",
}


def parse_line(line):
    """Parse '1: question What is...' -> (num, label, text)"""
    match = re.match(r"^(\d+):\s+(\w+)\s+(.+)$", line.strip())
    if match:
        return int(match.group(1)), match.group(2), match.group(3)
    return None


def format_line(num, label, text):
    """Format back to '1: question text'"""
    return f"{num}: {label} {text}"


def translate_batch(texts, from_code="en", to_code="es"):
    """Translate a batch of texts using argostranslate."""
    try:
        from argostranslate.translate import get_translation_from_codes

        translations = []
        for text in texts:
            try:
                translation = get_translation_from_codes(from_code, to_code)
                if translation:
                    result = translation.translate(text)
                    translations.append(result)
                else:
                    translations.append(text)
            except Exception as e:
                print(f"    Error: {e}")
                translations.append(text)
        return translations
    except ImportError:
        print("  argostranslate not installed")
        return None
    except Exception as e:
        print(f"  Translation error: {e}")
        return None


def main():
    base_path = Path(__file__).parent.parent / "train" / "clean_data"
    en_file = base_path / "sentence_types_EN.txt"

    # Read EN file
    with open(en_file) as f:
        lines = f.readlines()

    # Parse all
    entries = []
    for line in lines:
        parsed = parse_line(line)
        if parsed:
            entries.append(parsed)

    print(f"Loaded {len(entries)} entries from EN file")

    # Translate to each language in batches
    for lang, to_code in LANG_PAIRS.items():
        out_file = base_path / f"sentence_types_{lang.upper()}.txt"
        print(f"\nTranslating to {lang.upper()}...")

        # Translate in batches
        batch_size = 50
        texts = [e[2] for e in entries]
        all_translations = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            print(
                f"  {i + 1}-{min(i + batch_size, len(texts))}...", end=" ", flush=True
            )

            translated = translate_batch(batch, "en", to_code)
            if translated:
                all_translations.extend(translated)
                print("OK")
            else:
                all_translations.extend(batch)
                print("SKIP")

        # Write output
        with open(out_file, "w") as f:
            for entry, trans_text in zip(entries, all_translations):
                num, label, _ = entry
                f.write(format_line(num, label, trans_text) + "\n")

        print(f"  Wrote {len(entries)} lines to {out_file.name}")


if __name__ == "__main__":
    main()
