#!/usr/bin/env python3
"""Translate sentence types dataset while preserving labels using deep-translator."""

import re
from pathlib import Path
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def translate_text(text, lang):
    """Translate single text."""
    try:
        t = GoogleTranslator(source="en", target=lang)
        return t.translate(text)
    except Exception as e:
        return text


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

    # Translate to each language
    for lang in LANG_PAIRS:
        out_file = base_path / f"sentence_types_{lang.upper()}.txt"
        print(f"\nTranslating to {lang.upper()} ({len(entries)} sentences)...")

        texts = [e[2] for e in entries]

        # Use threading for speed
        translations = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(translate_text, text, lang): i
                for i, text in enumerate(texts)
            }

            done = 0
            for future in as_completed(futures):
                i = futures[future]
                try:
                    translations[i] = future.result()
                except Exception as e:
                    translations[i] = texts[i]
                done += 1
                if done % 500 == 0:
                    print(f"  {done}/{len(texts)} done")

        # Fill any None values with original
        translations = [t if t else texts[i] for i, t in enumerate(translations)]

        # Write output
        with open(out_file, "w") as f:
            for entry, trans_text in zip(entries, translations):
                num, label, _ = entry
                f.write(format_line(num, label, trans_text) + "\n")

        print(f"  Done! Wrote to {out_file.name}")


if __name__ == "__main__":
    main()
