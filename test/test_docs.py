"""The docs match the committed results and the code.

- every python block with a `print(...)  # value` line runs in a fresh process, after the imports of the
  earlier blocks on its page, and prints each value in order;
- the EAT figure data equals a fresh read of train/reports/eat/*_benchmark.json;
- the figure alt texts and the classification.md tables equal those reports.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAGES = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
PRINTED = re.compile(r"^\s*print\(.*\)\s+#\s?(.+?)\s*$")
sys.path.insert(0, str(ROOT / "docs" / "img"))
import plot_eat_models  # noqa: E402


def blocks():
    for page in PAGES:
        imports = []
        for n, code in enumerate(re.findall(r"^```python\n(.*?)^```", page.read_text(encoding="utf-8"), re.S | re.M)):
            expected = [m[1] for line in code.splitlines() if (m := PRINTED.match(line))]
            if expected:
                yield pytest.param("\n".join(imports) + "\n" + code, expected, id=f"{page.relative_to(ROOT)}-{n}",
                                   marks=known(code))
            imports += [line for line in code.splitlines() if re.match(r"(from \S+ )?import ", line)]


KNOWN = {  # text in the block: the open issue that makes the example fail
    'punctuated=False': "#78: onnxruntime rejects the unpunctuated EAT tokenizer pattern",
    '"Yes, they can perceive some colors."': "#77: the sentence-type model classifies answers as Question or Command",
    '"Yes, absolutely."': "#77: the sentence-type model classifies answers as Question or Command",
    'lang="fr"': "#79: no French EAT model, so a French Sentence fails",
}


def known(code):
    return [pytest.mark.xfail(strict=True, reason=why) for text, why in KNOWN.items() if text in code][:1]


@pytest.mark.parametrize("code,expected", list(blocks()))
def test_example_prints_its_comments(code, expected, tmp_path):
    done = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == expected


def test_figure_data_matches_the_reports():
    stem = ROOT / "docs" / "img" / "eat-models"
    committed = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    assert json.loads(json.dumps(plot_eat_models.load())) == committed


def test_alt_texts_match_the_reports():
    data = plot_eat_models.load()
    assert f'alt="{plot_eat_models.alt(data)}"' in (ROOT / "docs/classification.md").read_text(encoding="utf-8")
    assert f'alt="{plot_eat_models.alt(data, short=True)}"' in (ROOT / "docs/index.md").read_text(encoding="utf-8")


def test_classification_tables_match_the_reports():
    page = (ROOT / "docs/classification.md").read_text(encoding="utf-8")
    labels = {"two-stage default": "**Two-stage default**", "m2v 32M + tfidf": "`m2v-potion-base-32M+tfidf`",
              "eat53_svm": "`eat53_svm` (uncalibrated)", "eat53_svm_cal": "`eat53_svm_cal` (calibrated SVM)",
              "eat53_logreg": "`eat53_logreg`", "eat53_sgd": "`eat53_sgd`"}
    table = {c[0]: c[1:] for line in page.splitlines() if line.startswith("| ")
             for c in [[x.strip() for x in line.strip("|").split("|")]]}
    for label, acc, f1 in plot_eat_models.load()["rows"]:
        cells = [c.strip("*") for c in table[labels[label]]]
        assert cells == ["53", f"{acc:.1%}", f"{f1:.1%}"], label
    top = plot_eat_models.load()["rows"][0]
    assert [c.strip("*") for c in table["Accuracy"]] == [f"{top[1]:.1%}"]
    assert [c.strip("*") for c in table["Macro F1"]] == [f"{top[2]:.1%}"]
