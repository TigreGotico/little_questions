#!/usr/bin/env python3
"""Plot TF-IDF vs TF-IDF+categorical F1 comparison across languages."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DATA = {
    "DE": {"tfidf": 0.92, "categorical": 0.96},
    "EN": {"tfidf": 0.95, "categorical": 0.96},
    "ES": {"tfidf": 0.90, "categorical": 0.95},
    "FR": {"tfidf": 0.94, "categorical": 0.94},
    "IT": {"tfidf": 0.90, "categorical": 0.94},
    "NL": {"tfidf": 0.93, "categorical": 0.93},
    "PT": {"tfidf": 0.90, "categorical": 0.94},
}

langs = list(DATA.keys())
tfidf_vals = [DATA[l]["tfidf"] for l in langs]
cat_vals = [DATA[l]["categorical"] for l in langs]
deltas = [DATA[l]["categorical"] - DATA[l]["tfidf"] for l in langs]

x = np.arange(len(langs))
width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Categorical Feature Impact on Sentence-Type Classification (v0.8.0)", fontsize=13, y=1.01)

# ---- Left: grouped bar chart ----
bars1 = ax1.bar(x - width/2, tfidf_vals, width, label="TF-IDF only", color="#5b8db8", zorder=3)
bars2 = ax1.bar(x + width/2, cat_vals, width, label="TF-IDF + categorical", color="#e07b39", zorder=3)

ax1.set_ylabel("Macro F1")
ax1.set_title("Macro F1 by Language")
ax1.set_xticks(x)
ax1.set_xticklabels(langs)
ax1.set_ylim(0.85, 1.0)
ax1.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
ax1.set_axisbelow(True)
ax1.legend()

for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8, color="#5b8db8")
for bar in bars2:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8, color="#e07b39")

# ---- Right: delta bar chart ----
colors = ["#2ca02c" if d > 0 else "#d62728" if d < 0 else "#aaaaaa" for d in deltas]
bars3 = ax2.bar(x, deltas, color=colors, zorder=3)
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_ylabel("ΔF1 (categorical − TF-IDF)")
ax2.set_title("F1 Delta from Categorical Features")
ax2.set_xticks(x)
ax2.set_xticklabels(langs)
ax2.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
ax2.set_axisbelow(True)

for bar, delta in zip(bars3, deltas):
    offset = 0.001 if delta >= 0 else -0.003
    va = "bottom" if delta >= 0 else "top"
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
             f"{delta:+.2f}", ha="center", va=va, fontsize=9, fontweight="bold")

gain_patch = mpatches.Patch(color="#2ca02c", label="Gain")
flat_patch = mpatches.Patch(color="#aaaaaa", label="No change")
ax2.legend(handles=[gain_patch, flat_patch])

plt.tight_layout()
out = "train/reports/categorical_impact_0.8.0.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
