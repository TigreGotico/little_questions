#!/usr/bin/env python3
"""Generate polished benchmark visualisations from train/reports/*.txt.

Produces four figures saved to train/reports/:
  benchmark_cosc_svm_langs.png       — COSC 6-class TF-IDF SVM across 8 languages
  benchmark_cosc_en_models.png       — EN model-family showdown (SVM + all Potion variants)
  benchmark_sentence_type.png        — Sentence-type classifier across 8 languages
  benchmark_multilingual.png         — potion-multilingual-128M(+tfidf) across 8 languages
  benchmark_summary.png              — Combined 2×2 summary figure

Usage::

    python -m train.benchmark_plots          # saves to train/reports/
    python -m train.benchmark_plots --show   # also opens figures in a window
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPORTS = Path(__file__).resolve().parent / "reports"

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    "svm":          "#4E79A7",
    "2M":           "#F28E2B",
    "2M+tfidf":     "#E15759",
    "8M":           "#76B7B2",
    "8M+tfidf":     "#59A14F",
    "32M":          "#EDC948",
    "32M+tfidf":    "#B07AA1",
    "128M":         "#FF9DA7",
    "128M+tfidf":   "#9C755F",
    "acc":          "#4E79A7",
    "f1":           "#F28E2B",
}

LANG_LABELS = {
    "en": "English",
    "es": "Spanish",
    "pt": "Portuguese",
    "ca": "Catalan",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "nl": "Dutch",
}
LANGS = list(LANG_LABELS.keys())


# ── Parsers ───────────────────────────────────────────────────────────────────

def _parse_acc_f1(path: Path) -> tuple[float, float] | None:
    """Return (accuracy, macro_f1) from a report file."""
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    # Explicit summary lines: "Accuracy : 0.8775"
    acc_m = re.search(r"Accuracy\s*:\s*([\d.]+)", text)
    f1_m  = re.search(r"Macro F1\s*:\s*([\d.]+)", text)
    if acc_m and f1_m:
        return float(acc_m.group(1)), float(f1_m.group(1))
    # sklearn classification_report format: "    accuracy                0.9268  702"
    # accuracy row has only 2 numeric fields (score + support)
    acc_m = re.search(r"^\s+accuracy\s+([\d.]+)\s+\d+", text, re.MULTILINE)
    f1_m  = re.search(r"^\s+macro avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)", text, re.MULTILINE)
    if acc_m and f1_m:
        return float(acc_m.group(1)), float(f1_m.group(1))
    return None


def _parse_per_class_f1(path: Path) -> dict[str, float]:
    """Return {class_label: f1_score} from a classification_report file."""
    if not path.exists():
        return {}
    out: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        # "        ABBR       1.00      0.76      0.86        21"
        m = re.match(r"^\s+(\S+)\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+\d+", line)
        if m and m.group(1) not in ("accuracy", "macro", "weighted"):
            out[m.group(1)] = float(m.group(2))
    return out


# ── Styling helpers ───────────────────────────────────────────────────────────

def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "#FAFAFA",
        "axes.facecolor":   "#F4F4F4",
        "axes.edgecolor":   "#BBBBBB",
        "axes.grid":        True,
        "grid.color":       "#FFFFFF",
        "grid.linewidth":   1.0,
        "font.family":      "DejaVu Sans",
        "font.size":        11,
        "axes.titlesize":   13,
        "axes.labelsize":   11,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
        "legend.fontsize":  10,
        "legend.framealpha": 0.85,
    })


def _bar_label(ax, bars, fmt="{:.3f}", offset=0.003, fontsize=8.5):
    for bar in bars:
        v = bar.get_width()
        ax.text(
            v + offset, bar.get_y() + bar.get_height() / 2,
            fmt.format(v), va="center", ha="left",
            fontsize=fontsize, color="#333333",
        )


def _save(fig: plt.Figure, name: str, show: bool) -> Path:
    out = REPORTS / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  saved → {out}")
    if show:
        plt.show()
    plt.close(fig)
    return out


# ── Figure 1: COSC SVM across languages ──────────────────────────────────────

def fig_cosc_svm_langs(show: bool = False) -> Path:
    data: dict[str, tuple[float, float]] = {}
    for lang in LANGS:
        r = _parse_acc_f1(REPORTS / f"svm_{lang}_6c.txt")
        if r:
            data[lang] = r

    langs = list(data.keys())
    accs = [data[l][0] for l in langs]
    f1s  = [data[l][1] for l in langs]
    labels = [LANG_LABELS[l] for l in langs]
    y = np.arange(len(langs))
    h = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.barh(y + h/2, accs, h, color=PALETTE["acc"], label="Accuracy")
    b2 = ax.barh(y - h/2, f1s,  h, color=PALETTE["f1"],  label="Macro F1")
    _bar_label(ax, b1)
    _bar_label(ax, b2)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.70, 1.00)
    ax.set_xlabel("Score")
    ax.set_title("COSC 6-class  ·  TF-IDF + LinearSVC  ·  All Languages")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, "benchmark_cosc_svm_langs.png", show)


# ── Figure 2: EN model showdown ───────────────────────────────────────────────

def _model_key(name: str) -> str:
    """Short display name for a model variant."""
    if name == "svm":
        return "TF-IDF SVM"
    m = re.match(r"m2v-potion-(?:base-)?([\w.]+)", name)
    if m:
        return f"Potion {m.group(1)}"
    return name


def fig_en_model_showdown(show: bool = False) -> Path:
    models: list[tuple[str, float, float]] = []

    # TF-IDF SVM baseline
    r = _parse_acc_f1(REPORTS / "svm_en_6c.txt")
    if r:
        models.append(("TF-IDF SVM", r[0], r[1]))

    # m2v variants for EN
    for p in sorted(REPORTS.glob("m2v-potion-*-en-6c.txt")):
        r = _parse_acc_f1(p)
        if r:
            # derive display name
            stem = p.stem  # m2v-potion-base-8M+tfidf-en-6c
            tag = stem.replace("-en-6c", "").replace("m2v-potion-base-", "")
            tag = tag.replace("m2v-potion-multilingual-", "multilingual-")
            models.append((tag, r[0], r[1]))

    # Sort by Macro F1 descending
    models.sort(key=lambda x: x[2])

    names = [m[0] for m in models]
    accs  = [m[1] for m in models]
    f1s   = [m[2] for m in models]
    y = np.arange(len(names))
    h = 0.35

    fig, ax = plt.subplots(figsize=(11, max(5, len(names) * 0.55)))
    b1 = ax.barh(y + h/2, accs, h, color=PALETTE["acc"], label="Accuracy")
    b2 = ax.barh(y - h/2, f1s,  h, color=PALETTE["f1"],  label="Macro F1")
    _bar_label(ax, b1)
    _bar_label(ax, b2)

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0.70, 1.00)
    ax.set_xlabel("Score")
    ax.set_title("COSC 6-class  ·  EN  ·  Model Family Comparison  (ranked by Macro F1)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, "benchmark_cosc_en_models.png", show)


# ── Figure 3: Sentence-type across languages ──────────────────────────────────

def fig_sentence_type(show: bool = False) -> Path:
    data: dict[str, tuple[float, float]] = {}
    for lang in LANGS:
        sfx = lang.upper()
        r = _parse_acc_f1(REPORTS / f"sentence_type_{sfx}_0.8.0.txt")
        if r:
            data[lang] = r

    langs = list(data.keys())
    accs = [data[l][0] for l in langs]
    f1s  = [data[l][1] for l in langs]
    labels = [LANG_LABELS[l] for l in langs]
    y = np.arange(len(langs))
    h = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.barh(y + h/2, accs, h, color="#59A14F", label="Accuracy")
    b2 = ax.barh(y - h/2, f1s,  h, color="#F28E2B", label="Macro F1")
    _bar_label(ax, b1)
    _bar_label(ax, b2)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.70, 1.00)
    ax.set_xlabel("Score")
    ax.set_title("Sentence-Type Classifier  ·  TF-IDF + LinearSVC  ·  All Languages\n"
                 "(5 classes: question / statement / command / exclamation / request)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, "benchmark_sentence_type.png", show)


# ── Figure 4: Multilingual Potion across languages ────────────────────────────

def fig_multilingual(show: bool = False) -> Path:
    base_data:  dict[str, tuple[float, float]] = {}
    fused_data: dict[str, tuple[float, float]] = {}
    svm_data:   dict[str, tuple[float, float]] = {}

    for lang in LANGS:
        r = _parse_acc_f1(REPORTS / f"m2v-potion-multilingual-128M-{lang}-6c.txt")
        if r:
            base_data[lang] = r
        r = _parse_acc_f1(REPORTS / f"m2v-potion-multilingual-128M+tfidf-{lang}-6c.txt")
        if r:
            fused_data[lang] = r
        r = _parse_acc_f1(REPORTS / f"svm_{lang}_6c.txt")
        if r:
            svm_data[lang] = r

    # Use union of all langs that have any data
    langs = [l for l in LANGS if l in base_data or l in fused_data]
    labels = [LANG_LABELS[l] for l in langs]
    y = np.arange(len(langs))
    h = 0.25

    fig, ax = plt.subplots(figsize=(11, 5.5))

    def _vals(d, langs, idx=1):
        return [d[l][idx] if l in d else 0 for l in langs]

    b1 = ax.barh(y + h,    _vals(svm_data,   langs, 1), h, color=PALETTE["acc"],       label="TF-IDF SVM (Macro F1)")
    b2 = ax.barh(y,        _vals(base_data,  langs, 1), h, color=PALETTE["128M"],      label="Potion-128M (Macro F1)")
    b3 = ax.barh(y - h,    _vals(fused_data, langs, 1), h, color=PALETTE["128M+tfidf"],label="Potion-128M + TF-IDF (Macro F1)")

    for bars in (b1, b2, b3):
        _bar_label(ax, bars, offset=0.002)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.65, 1.00)
    ax.set_xlabel("Macro F1")
    ax.set_title("COSC 6-class  ·  Multilingual Potion vs TF-IDF SVM  ·  All Languages")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, "benchmark_multilingual.png", show)


# ── Figure 5: Per-class heatmap for SVM EN 6c ────────────────────────────────

def fig_per_class_heatmap(show: bool = False) -> Path:
    """F1 heatmap across languages and COSC 6 classes."""
    classes = ["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"]
    matrix = np.zeros((len(LANGS), len(classes)))

    for i, lang in enumerate(LANGS):
        pc = _parse_per_class_f1(REPORTS / f"svm_{lang}_6c.txt")
        for j, cls in enumerate(classes):
            matrix[i, j] = pc.get(cls, np.nan)

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticks(range(len(LANGS)))
    ax.set_yticklabels([LANG_LABELS[l] for l in LANGS], fontsize=11)
    ax.set_title("COSC 6-class  ·  TF-IDF SVM  ·  Per-class F1  (by language)")

    for i in range(len(LANGS)):
        for j in range(len(classes)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, color="black" if v > 0.6 else "white")

    fig.colorbar(im, ax=ax, label="F1 score", fraction=0.03, pad=0.02)
    fig.tight_layout()
    return _save(fig, "benchmark_per_class_heatmap.png", show)


# ── Figure 6: 2×2 summary ─────────────────────────────────────────────────────

def fig_summary(show: bool = False) -> Path:
    """2×2 composite of all four main benchmarks."""
    _style()
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("little-questions  ·  Benchmark Summary  ·  v0.8.0",
                 fontsize=16, fontweight="bold", y=0.98)

    axes = fig.subplots(2, 2)

    def _draw_cosc_svm(ax):
        data = {l: _parse_acc_f1(REPORTS / f"svm_{l}_6c.txt") for l in LANGS}
        data = {k: v for k, v in data.items() if v}
        langs = list(data.keys()); y = np.arange(len(langs)); h = 0.35
        b1 = ax.barh(y+h/2, [data[l][0] for l in langs], h, color=PALETTE["acc"], label="Accuracy")
        b2 = ax.barh(y-h/2, [data[l][1] for l in langs], h, color=PALETTE["f1"],  label="Macro F1")
        ax.set_yticks(y); ax.set_yticklabels([LANG_LABELS[l] for l in langs])
        ax.set_xlim(0.70, 1.00); ax.set_title("COSC 6-class · TF-IDF SVM"); ax.legend(fontsize=8)

    def _draw_en_models(ax):
        models = []
        r = _parse_acc_f1(REPORTS / "svm_en_6c.txt")
        if r: models.append(("TF-IDF SVM", r[0], r[1]))
        for p in sorted(REPORTS.glob("m2v-potion-*-en-6c.txt")):
            r = _parse_acc_f1(p)
            if r:
                tag = p.stem.replace("-en-6c","").replace("m2v-potion-base-","").replace("m2v-potion-multilingual-","ml-")
                models.append((tag, r[0], r[1]))
        models.sort(key=lambda x: x[2])
        names=[m[0] for m in models]; accs=[m[1] for m in models]; f1s=[m[2] for m in models]
        y=np.arange(len(names)); h=0.35
        ax.barh(y+h/2, accs, h, color=PALETTE["acc"], label="Accuracy")
        ax.barh(y-h/2, f1s,  h, color=PALETTE["f1"],  label="Macro F1")
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
        ax.set_xlim(0.70, 1.00); ax.set_title("EN · Model Comparison"); ax.legend(fontsize=8)

    def _draw_sentence_type(ax):
        data = {}
        for lang in LANGS:
            r = _parse_acc_f1(REPORTS / f"sentence_type_{lang.upper()}_0.8.0.txt")
            if r: data[lang] = r
        langs=list(data.keys()); y=np.arange(len(langs)); h=0.35
        ax.barh(y+h/2, [data[l][0] for l in langs], h, color="#59A14F", label="Accuracy")
        ax.barh(y-h/2, [data[l][1] for l in langs], h, color="#F28E2B", label="Macro F1")
        ax.set_yticks(y); ax.set_yticklabels([LANG_LABELS[l] for l in langs])
        ax.set_xlim(0.70, 1.00); ax.set_title("Sentence-Type Classifier"); ax.legend(fontsize=8)

    def _draw_multilingual(ax):
        svm={l:_parse_acc_f1(REPORTS/f"svm_{l}_6c.txt") for l in LANGS}
        fused={l:_parse_acc_f1(REPORTS/f"m2v-potion-multilingual-128M+tfidf-{l}-6c.txt") for l in LANGS}
        svm={k:v for k,v in svm.items() if v}; fused={k:v for k,v in fused.items() if v}
        langs=[l for l in LANGS if l in fused]
        y=np.arange(len(langs)); h=0.3
        ax.barh(y+h/2, [svm.get(l,(0,0))[1] for l in langs],   h, color=PALETTE["acc"],      label="TF-IDF SVM")
        ax.barh(y-h/2, [fused.get(l,(0,0))[1] for l in langs], h, color=PALETTE["128M+tfidf"],label="Potion-128M+TF-IDF")
        ax.set_yticks(y); ax.set_yticklabels([LANG_LABELS[l] for l in langs])
        ax.set_xlim(0.65, 1.00); ax.set_title("Multilingual Potion vs SVM"); ax.legend(fontsize=8)

    _draw_cosc_svm(axes[0, 0])
    _draw_en_models(axes[0, 1])
    _draw_sentence_type(axes[1, 0])
    _draw_multilingual(axes[1, 1])

    for ax in axes.flat:
        ax.set_xlabel("Score")
        ax.grid(True, axis="x", color="white", linewidth=0.8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, "benchmark_summary.png", show)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark plots")
    parser.add_argument("--show", action="store_true", help="Show figures interactively")
    args = parser.parse_args()

    _style()
    print("Generating benchmark plots …")
    fig_cosc_svm_langs(args.show)
    fig_en_model_showdown(args.show)
    fig_sentence_type(args.show)
    fig_multilingual(args.show)
    fig_per_class_heatmap(args.show)
    fig_summary(args.show)
    print("\nAll plots written to train/reports/")


if __name__ == "__main__":
    main()
