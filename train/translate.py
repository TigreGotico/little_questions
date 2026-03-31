#!/usr/bin/env python3
"""Translate training data to new languages."""

import argparse
import os
import sys
from os.path import join, dirname

sys.path.insert(0, join(dirname(__file__), ".."))

import argostranslate.translate

DATA_DIR = join(dirname(__file__), "clean_data")
SOURCE_FILE = "raw_questions_0.7.0a1.txt"

LANG_MAP = {
    "de": "German",
    "it": "Italian",
    "nl": "Dutch",
}


def translate_text(text, from_lang="en", to_lang="de"):
    """Translate text using argostranslate."""
    try:
        translated = argostranslate.translate.translate(text, from_lang, to_lang)
        return translated
    except Exception as e:
        print(f"Translation error: {e}")
        return None


def translate_dataset(target_lang, batch_size=50):
    """Translate the English dataset to target language."""
    target_name = LANG_MAP.get(target_lang)
    if not target_name:
        print(f"Unknown language: {target_lang}")
        return False

    print(f"Translating to {target_name} ({target_lang})...")

    source_path = join(DATA_DIR, SOURCE_FILE)
    if not os.path.exists(source_path):
        print(f"Source file not found: {source_path}")
        return False

    with open(source_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Translating {len(lines)} questions...")

    output_lines = []
    translated_count = 0

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue

        label = parts[0]
        text = parts[1]

        translated = translate_text(text, "en", target_lang)

        if translated:
            output_lines.append(f"{label} {translated}\n")
            translated_count += 1
        else:
            output_lines.append(f"{label} {text}\n")

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(lines)} ({translated_count} translated)")

        if batch_size > 0 and (i + 1) % batch_size == 0:
            import time

            time.sleep(0.1)

    output_path = join(DATA_DIR, f"raw_questions_{target_lang.upper()}_0.8.0.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(output_lines)

    print(f"Saved {translated_count}/{len(lines)} translations to {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Translate training data")
    parser.add_argument("lang", help="Target language code (de, it, nl)")
    parser.add_argument(
        "--batch-size", type=int, default=50, help="Batch size for progress updates"
    )
    args = parser.parse_args()

    success = translate_dataset(args.lang, args.batch_size)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
