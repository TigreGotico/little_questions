"""Draw docs/img/eat-models.svg: accuracy and macro F1 of each EAT model on the held-out split.

Run from the repo root: python docs/img/plot_eat_models.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from plot_style import save  # noqa: E402

REPORTS = ROOT / "train" / "reports" / "eat"
MODELS = {
    "eat53_2stage_svm_cal": "two-stage default",
    "eat53_svm_cal_EN_0.9.0": "eat53_svm_cal",
    "eat53_svm_EN_0.9.0": "eat53_svm",
    "m2v-potion-base-32M+tfidf-en-53c": "m2v 32M + tfidf",
    "eat53_logreg_EN_0.9.0": "eat53_logreg",
    "eat53_sgd_EN_0.9.0": "eat53_sgd",
}
rows = []
for scorer, label in MODELS.items():
    d = json.loads((REPORTS / f"{scorer}_benchmark.json").read_text(encoding="utf-8"))
    rows.append((label, d["accuracy"], d["macro_f1"]))
rows.sort(key=lambda r: -r[1])


def draw(fig, theme):
    ax = fig.subplots()
    y = range(len(rows))
    ax.scatter([r[1] for r in rows], list(y), s=40, color=theme["series"][0], label="accuracy", zorder=3)
    ax.scatter([r[2] for r in rows], list(y), s=40, color=theme["series"][1], marker="D", label="macro F1", zorder=3)
    for i, r in zip(y, rows):
        ax.text(max(r[1], r[2]) + 0.004, i, f"{r[1]:.1%} / {r[2]:.1%}", va="center", fontsize=8, color=theme["text"])
    ax.set_yticks(list(y), [r[0] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.8, 1.0)
    ax.set_xticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("EAT test split, 53 classes, 4,503 questions")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8)


save(str(ROOT / "docs" / "img" / "eat-models"), draw, width=7, height=3.4)
