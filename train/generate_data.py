#!/usr/bin/env python3
"""Generate synthetic training data using Qwen LLM."""

import argparse
import os
import sys
import requests
from os.path import join, dirname

DATA_DIR = join(dirname(__file__), "clean_data")
LLM_URL = "http://192.168.1.200:8000"

COSC_LABELS = [
    "ABBR:abb",
    "ABBR:exp",
    "DESC:def",
    "DESC:desc",
    "DESC:manner",
    "DESC:reason",
    "ENTY:animal",
    "ENTY:body",
    "ENTY:color",
    "ENTY:cremat",
    "ENTY:currency",
    "ENTY:dismed",
    "ENTY:event",
    "ENTY:food",
    "ENTY:instru",
    "ENTY:lang",
    "ENTY:letter",
    "ENTY:other",
    "ENTY:plant",
    "ENTY:product",
    "ENTY:religion",
    "ENTY:sport",
    "ENTY:substance",
    "ENTY:symbol",
    "ENTY:techmeth",
    "ENTY:termeq",
    "ENTY:veh",
    "ENTY:word",
    "HUM:desc",
    "HUM:gr",
    "HUM:ind",
    "HUM:title",
    "LOC:city",
    "LOC:country",
    "LOC:landmass",
    "LOC:mount",
    "LOC:other",
    "LOC:state",
    "LOC:water",
    "NUM:code",
    "NUM:count",
    "NUM:date",
    "NUM:dist",
    "NUM:money",
    "NUM:ord",
    "NUM:other",
    "NUM:perc",
    "NUM:period",
    "NUM:speed",
    "NUM:temp",
    "NUM:volsize",
    "NUM:weight",
]


def generate_questions(label: str, lang: str, count: int = 10) -> list:
    """Generate questions for a single label."""
    prompt = f"""Generate {count} natural questions in English that expect a '{label}' type answer.
Format: Start each line with exactly "{label} "

Example: HUM:ind Who created Python?

Generate {count} lines:\n"""

    try:
        response = requests.post(
            f"{LLM_URL}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.8,
                "max_tokens": 500,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]

        questions = []
        for line in content.strip().split("\n"):
            line = line.strip()
            if line.startswith(label + " "):
                q = line[len(label) :].strip()
                if q and len(q) > 5:
                    questions.append(f"{label} {q}")

        return questions[:count]
    except Exception as e:
        print(f"Error for {label}: {e}")
        return []


def augment_dataset(lang: str, output_file: str, samples_per_label: int = 50):
    """Augment dataset."""
    print(f"\nGenerating {samples_per_label} samples x {len(COSC_LABELS)} labels...")

    all_questions = []

    for i, label in enumerate(COSC_LABELS):
        qs = generate_questions(label, lang, samples_per_label)
        all_questions.extend(qs)
        print(f"[{i + 1}/{len(COSC_LABELS)}] {label}: {len(qs)}")

    output_path = join(DATA_DIR, output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_questions))

    print(f"\nSaved {len(all_questions)} questions to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="en")
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()

    output_file = f"raw_questions_{args.lang.upper()}_aug_0.8.0.txt"
    augment_dataset(args.lang, output_file, args.samples)


if __name__ == "__main__":
    main()
