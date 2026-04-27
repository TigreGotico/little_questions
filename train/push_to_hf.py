#!/usr/bin/env python3
"""Push all trained EAT models to a single HuggingFace repo.

Repo: TigreGotico/eat-classifiers

Layout uploaded:
  models/eat/eat53_svm_EN_0.9.0.onnx
  models/eat/eat7_svm_EN_0.9.0.onnx
  ... (all ONNX files)
  models/eat_m2v/m2v-potion-base-8M+tfidf-en-53c/
  ... (all M2V directories)
  benchmarks/eat/benchmark_eat53_overview.png
  ... (all benchmark plots and JSON reports)
  README.md
  BENCHMARKS.md

Usage::

    python -m train.push_to_hf                   # push everything
    python -m train.push_to_hf --dry-run         # print what would be uploaded
    python -m train.push_to_hf --type onnx       # only ONNX models
    python -m train.push_to_hf --type m2v        # only M2V models
    python -m train.push_to_hf --type reports    # only benchmark reports + plots
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from textwrap import dedent

LOG = logging.getLogger(__name__)

HF_REPO_ID = "TigreGotico/eat-classifiers"
ONNX_MODEL_DIR = Path(os.path.expanduser("~/.local/share/little_questions/eat"))
M2V_MODEL_DIR = Path(os.path.expanduser("~/.local/share/little_questions/eat_m2v"))
REPORTS_DIR = Path(__file__).parent / "reports" / "eat"
VERSION = "0.9.0"

EAT_DATASET_URL = "https://huggingface.co/datasets/TigreGotico/EAT"
SENTENCE_TYPES_DATASET_URL = "https://huggingface.co/datasets/TigreGotico/sentence-types-multilingual"
LITTLE_QUESTIONS_URL = "https://github.com/TigreGotico/little_questions"


# ---------------------------------------------------------------------------
# README / BENCHMARKS generators
# ---------------------------------------------------------------------------

def _load_benchmark_scores() -> dict[str, dict]:
    """Return {scorer_name: {accuracy, macro_f1, weighted_f1, n_labels}} from JSON reports."""
    scores = {}
    if not REPORTS_DIR.exists():
        return scores
    for rf in REPORTS_DIR.glob("*_benchmark.json"):
        try:
            data = json.loads(rf.read_text())
            name = data.get("scorer_name", rf.stem.replace("_benchmark", ""))
            scores[name] = {
                "accuracy": data.get("accuracy", 0),
                "macro_f1": data.get("macro_f1", 0),
                "weighted_f1": data.get("weighted_f1", 0),
                "n_labels": len(data.get("labels", [])),
            }
        except Exception:
            pass
    return scores


def _generate_readme(onnx_files: list[Path], m2v_dirs: list[Path]) -> str:
    scores = _load_benchmark_scores()

    # Build ONNX table with benchmark numbers
    onnx_rows = []
    for f in sorted(onnx_files):
        stem = f.stem
        parts = stem.split("_")
        n_cls = parts[0].replace("eat", "")
        model_type = "_".join(parts[1:-2])
        calibrated = "cal" in model_type
        prob_note = "calibrated proba" if calibrated else "decision scores"
        s = scores.get(stem, {})
        acc = f"{s['accuracy']:.3f}" if s else "—"
        mf1 = f"{s['macro_f1']:.3f}" if s else "—"
        onnx_rows.append(
            f"| `{f.name}` | {n_cls} | {model_type} | {prob_note} | {acc} | {mf1} |"
        )
    # Add 2-stage rows
    for mt in ("svm", "svm_cal"):
        key = f"eat53_2stage_{mt}"
        s = scores.get(key, {})
        acc = f"{s['accuracy']:.3f}" if s else "—"
        mf1 = f"{s['macro_f1']:.3f}" if s else "—"
        calibrated = "cal" in mt
        prob_note = "calibrated proba" if calibrated else "decision scores"
        onnx_rows.append(
            f"| _(two-stage {mt})_ | 53 | {mt} | {prob_note} | {acc} | {mf1} |"
        )

    # Build M2V table with benchmark numbers
    m2v_rows = []
    for d in sorted(m2v_dirs):
        meta_path = d / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            n_cls = len(meta.get("classes", []))
            backbone = meta.get("model_name", "").split("/")[-1]
            fusion = "✓" if meta.get("use_tfidf_fusion") else "—"
        else:
            n_cls, backbone, fusion = "?", d.name, "?"
        s = scores.get(d.name, {})
        acc = f"{s['accuracy']:.3f}" if s else "—"
        mf1 = f"{s['macro_f1']:.3f}" if s else "—"
        m2v_rows.append(
            f"| `{d.name}` | {n_cls} | {backbone} | {fusion} | {acc} | {mf1} |"
        )

    onnx_table = "\n".join(onnx_rows) or "_No ONNX models found_"
    m2v_table = "\n".join(m2v_rows) or "_No M2V models found_"

    return dedent(f"""\
        ---
        language:
          - en
        license: apache-2.0
        tags:
          - question-classification
          - text-classification
          - onnx
          - model2vec
          - english
          - eat
          - calibrated
        datasets:
          - TigreGotico/EAT
        ---

        # eat-classifiers

        English question-type classifiers trained on the
        [EAT (Expected Answer Type) dataset]({EAT_DATASET_URL}) — 30,017 questions
        labelled with 53 fine-grained answer types across 7 main categories.

        Two model families are included:

        | Family | Location | Inference deps | Best macro F1 (53-class) |
        |--------|----------|----------------|--------------------------|
        | **ONNX** (LinearSVC, calibrated) | `models/eat/` | `onnxruntime` only | **0.934** (two-stage svm_cal) |
        | **Model2Vec** (potion-base + LinearSVC) | `models/eat_m2v/` | `model2vec`, `scikit-learn` | **0.921** (32M+tfidf) |

        ---

        ## Label taxonomy

        7 main categories and 53 fine-grained sub-types:

        | Main | Sub-types |
        |------|-----------|
        | `ABBR` | `abb`, `exp` |
        | `BOOL` | `yesno` |
        | `DESC` | `def`, `desc`, `manner`, `reason` |
        | `ENTY` | `animal`, `body`, `color`, `cremat`, `currency`, `dismed`, `event`, `food`, `instru`, `lang`, `letter`, `other`, `plant`, `product`, `religion`, `sport`, `substance`, `symbol`, `techmeth`, `termeq`, `veh`, `word` |
        | `HUM` | `desc`, `gr`, `ind`, `title` |
        | `LOC` | `city`, `country`, `landmass`, `mount`, `other`, `state`, `water` |
        | `NUM` | `code`, `count`, `date`, `dist`, `money`, `ord`, `other`, `perc`, `period`, `speed`, `temp`, `volsize`, `weight` |

        ---

        ## ONNX models

        Calibrated models (`_cal`) use Platt sigmoid calibration
        (`CalibratedClassifierCV(LinearSVC, cv=5, method="sigmoid")`) so that
        `output[1]` is a true probability vector — values in [0, 1] summing to 1.
        Uncalibrated models output raw decision-function scores.

        | File | Classes | Type | Output[1] | Accuracy | Macro F1 |
        |------|---------|------|-----------|----------|----------|
        {onnx_table}

        ### Two-stage inference

        Run `eat7_svm_cal` first to predict the main category, then run
        `eat53_svm_cal`, zero out all labels outside that category, and
        renormalise the surviving probabilities. This is the highest-accuracy
        configuration (macro F1 = 0.934 on the held-out test set).

        ```python
        import onnxruntime as rt
        import numpy as np, json

        sess7  = rt.InferenceSession("models/eat/eat7_svm_cal_EN_0.9.0.onnx")
        sess53 = rt.InferenceSession("models/eat/eat53_svm_cal_EN_0.9.0.onnx")
        classes7  = json.loads(sess7.get_modelmeta().custom_metadata_map["classes"])
        classes53 = json.loads(sess53.get_modelmeta().custom_metadata_map["classes"])
        main_of_53 = [c.split(":")[0] for c in classes53]

        def classify(text: str) -> tuple[str, float]:
            inp = np.array([text], dtype=object)
            main_idx = int(sess7.run(None, {{"input": inp}})[0][0])
            main = classes7[main_idx]
            _, probs53 = sess53.run(None, {{"input": inp}})
            row = probs53[0].copy()
            for j, m in enumerate(main_of_53):
                if m != main:
                    row[j] = 0.0
            row /= row.sum()
            best = int(np.argmax(row))
            return classes53[best], float(row[best])

        label, confidence = classify("Who invented the telephone?")
        print(label, confidence)   # HUM:ind  0.96
        ```

        ### Single-model ONNX inference

        ```python
        import onnxruntime as rt
        import numpy as np, json

        sess = rt.InferenceSession("models/eat/eat53_svm_cal_EN_0.9.0.onnx")
        classes = json.loads(sess.get_modelmeta().custom_metadata_map["classes"])

        inp = np.array(["Who invented the telephone?"], dtype=object)
        label_idx, probs = sess.run(None, {{"input": inp}})
        label = classes[int(label_idx[0])]
        confidence = float(probs[0].max())
        print(label, confidence)   # HUM:ind  0.94
        ```

        ---

        ## Model2Vec models

        Static [model2vec](https://github.com/MinishLab/model2vec) embeddings
        (potion-base 2M / 8M / 32M) with a LinearSVC classifier on top.
        The `+tfidf` variants concatenate TF-IDF word (1,2)-gram features with
        the embeddings before fitting — this consistently adds 5–30 pp macro F1
        over embeddings alone.

        | Directory | Classes | Backbone | TF-IDF fusion | Accuracy | Macro F1 |
        |-----------|---------|----------|---------------|----------|----------|
        {m2v_table}

        ### Usage (Model2Vec)

        ```python
        from huggingface_hub import snapshot_download
        from train.classifiers import Model2VecClassifier

        local = snapshot_download("TigreGotico/eat-classifiers")
        clf = Model2VecClassifier.load(f"{{local}}/models/eat_m2v/m2v-potion-base-8M+tfidf-en-53c")
        print(clf.predict(["Who invented the telephone?"]))
        # ["HUM:ind"]
        ```

        ---

        ## Integration with little_questions

        When installed as part of [little_questions]({LITTLE_QUESTIONS_URL}), the
        calibrated ONNX model (`eat53_svm_cal`) is loaded automatically and
        exposed through the `Sentence` API:

        ```python
        from little_questions import Sentence

        s = Sentence("Who invented the telephone?")
        print(s.classification)          # "HUM:ind"
        print(s.main_label)              # "HUM"
        print(s.secondary_label)         # "ind"
        print(s.confidence)              # 0.94
        print(s.pretty_label)            # "individual (Human)"
        # Full probability distribution over all 53 labels:
        print(s.classification_scores)
        ```

        ---

        ## Datasets

        Both datasets were built specifically for this project:

        | Dataset | Task | Size | Labels |
        |---------|------|------|--------|
        | [TigreGotico/EAT]({EAT_DATASET_URL}) | Question answer-type | 30K EN | 53 fine-grained / 7 main |
        | [TigreGotico/sentence-types-multilingual]({SENTENCE_TYPES_DATASET_URL}) | Sentence type | 80K multilingual | 5 types |

        ---

        ## Benchmarks

        Full results tables and confusion matrices: [BENCHMARKS.md](BENCHMARKS.md)

        ![Model comparison](benchmarks/eat/benchmark_eat_model_comparison.png)
        """)


def _generate_benchmarks() -> str:
    report_files = sorted(REPORTS_DIR.glob("*_benchmark.json")) if REPORTS_DIR.exists() else []
    rows_53, rows_7 = [], []
    for rf in report_files:
        try:
            data = json.loads(rf.read_text())
        except Exception:
            continue
        name = data.get("scorer_name", rf.stem.replace("_benchmark", ""))
        acc = data.get("accuracy", 0)
        mf1 = data.get("macro_f1", 0)
        wf1 = data.get("weighted_f1", 0)
        n_labels = len(data.get("labels", []))
        row = f"| `{name}` | {acc:.4f} | {mf1:.4f} | {wf1:.4f} |"
        if n_labels > 10:
            rows_53.append((mf1, row))
        else:
            rows_7.append((mf1, row))

    def table(rows):
        if not rows:
            return "_No results found_"
        header = "| Model | Accuracy | Macro F1 | Weighted F1 |\n|-------|----------|----------|-------------|"
        body = "\n".join(r for _, r in sorted(rows, reverse=True))
        return header + "\n" + body

    return dedent(f"""\
        # EAT Classifier Benchmarks

        Test set: 15% stratified split of [TigreGotico/EAT](https://huggingface.co/datasets/TigreGotico/EAT)
        (random_state=42), 4,503 samples.

        ## 53-class results

        {table(rows_53)}

        ## 7-class results

        {table(rows_7)}

        ## Plots

        ![Overview 53-class](benchmarks/eat/benchmark_eat53_overview.png)
        ![Overview 7-class](benchmarks/eat/benchmark_eat7_overview.png)
        ![Model comparison](benchmarks/eat/benchmark_eat_model_comparison.png)
        ![Per-class F1 53](benchmarks/eat/benchmark_eat53_per_class_f1.png)
        ![Per-class F1 7](benchmarks/eat/benchmark_eat7_per_class_f1.png)
        """)


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------

def push_all(push_onnx: bool, push_m2v: bool, push_reports: bool,
             dry_run: bool = False) -> None:
    from huggingface_hub import HfApi

    api = HfApi() if not dry_run else None

    def upload_file(local_path: Path, repo_path: str) -> None:
        print(f"  {'[dry] ' if dry_run else ''}upload {repo_path}")
        if dry_run:
            return
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=HF_REPO_ID,
            repo_type="model",
        )

    def upload_bytes(content: bytes, repo_path: str) -> None:
        print(f"  {'[dry] ' if dry_run else ''}upload {repo_path}")
        if dry_run:
            return
        api.upload_file(
            path_or_fileobj=content,
            path_in_repo=repo_path,
            repo_id=HF_REPO_ID,
            repo_type="model",
        )

    onnx_files = sorted(ONNX_MODEL_DIR.glob("*.onnx")) if ONNX_MODEL_DIR.exists() else []
    m2v_dirs = [d for d in sorted(M2V_MODEL_DIR.iterdir())
                if d.is_dir()] if M2V_MODEL_DIR.exists() else []

    if not dry_run:
        print(f"Creating repo {HF_REPO_ID} (exist_ok=True)…")
        api.create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True)

    # README + BENCHMARKS
    readme = _generate_readme(onnx_files, m2v_dirs)
    benchmarks = _generate_benchmarks()
    upload_bytes(readme.encode(), "README.md")
    upload_bytes(benchmarks.encode(), "BENCHMARKS.md")

    # ONNX models
    if push_onnx:
        print(f"\nUploading {len(onnx_files)} ONNX models…")
        for f in onnx_files:
            upload_file(f, f"models/eat/{f.name}")

    # M2V models
    if push_m2v:
        print(f"\nUploading {len(m2v_dirs)} M2V model directories…")
        for d in m2v_dirs:
            for fp in sorted(d.rglob("*")):
                if fp.is_file():
                    rel = fp.relative_to(M2V_MODEL_DIR)
                    upload_file(fp, f"models/eat_m2v/{rel}")

    # Benchmark reports + plots
    if push_reports and REPORTS_DIR.exists():
        report_files = list(REPORTS_DIR.glob("*.json")) + list(REPORTS_DIR.glob("*.png"))
        print(f"\nUploading {len(report_files)} benchmark reports/plots…")
        for rf in sorted(report_files):
            upload_file(rf, f"benchmarks/eat/{rf.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description=f"Push EAT models to {HF_REPO_ID}")
    parser.add_argument("--type", choices=["onnx", "m2v", "reports", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    push_all(
        push_onnx=args.type in ("onnx", "all"),
        push_m2v=args.type in ("m2v", "all"),
        push_reports=args.type in ("reports", "all"),
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        print(f"\nDone → https://huggingface.co/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
