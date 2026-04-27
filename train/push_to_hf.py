#!/usr/bin/env python3
"""Push trained EAT models to HuggingFace — one repo per model.

ONNX repos: TigreGotico/eat{N}_{model_type}_en
M2V repos:  TigreGotico/{variant_name}   (e.g. m2v-potion-base-8M+tfidf-en-53c)

Each repo receives:
  - the model file(s)
  - a generated README.md with YAML model card frontmatter

Usage::

    python -m train.push_to_hf --type onnx          # push all trained ONNX models
    python -m train.push_to_hf --type m2v           # push all trained M2V models
    python -m train.push_to_hf --type all           # push everything (default)
    python -m train.push_to_hf --model eat53_svm_cal_EN_0.9.0   # single model
    python -m train.push_to_hf --dry-run            # print repos that would be created
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from textwrap import dedent

LOG = logging.getLogger(__name__)

HF_ORG = "TigreGotico"
ONNX_MODEL_DIR = Path(os.path.expanduser("~/.local/share/little_questions/eat"))
M2V_MODEL_DIR = Path(os.path.expanduser("~/.local/share/little_questions/eat_m2v"))
VERSION = "0.9.0"

EAT_DATASET_URL = "https://huggingface.co/datasets/TigreGotico/EAT"
SENTENCE_TYPES_DATASET_URL = "https://huggingface.co/datasets/TigreGotico/sentence-types-multilingual"
LITTLE_QUESTIONS_URL = "https://github.com/TigreGotico/little_questions"


# ---------------------------------------------------------------------------
# README generators
# ---------------------------------------------------------------------------

def _onnx_readme(model_stem: str, classes: list[str], n_cls: int,
                 model_type: str, calibrated: bool) -> str:
    prob_note = (
        "Output `[1]` is a calibrated probability vector (Platt sigmoid, values in [0,1], sum to 1.0)."
        if calibrated else
        "Output `[1]` contains raw decision-function scores (not probabilities)."
    )
    tags = ["question-classification", "text-classification", "onnx", "english", "eat"]
    if calibrated:
        tags.append("calibrated")
    tags_yaml = "\n".join(f"  - {t}" for t in tags)

    classes_md = "\n".join(f"  - `{c}`" for c in classes)

    return dedent(f"""\
        ---
        language:
          - en
        license: apache-2.0
        tags:
        {tags_yaml}
        datasets:
          - TigreGotico/EAT
        ---

        # {model_stem}

        EAT question-type classifier — {n_cls} classes, `{model_type}` backbone.

        Trained on the [EAT dataset]({EAT_DATASET_URL}) (30K English questions, {n_cls} labels).
        Inference requires only `onnxruntime` — no sklearn at runtime.

        ## Usage

        ```python
        import onnxruntime as rt
        import numpy as np, json

        sess = rt.InferenceSession("{model_stem}.onnx")
        meta = sess.get_modelmeta().custom_metadata_map
        classes = json.loads(meta["classes"])

        inp = np.array(["Who invented the telephone?"], dtype=object)
        label_idx, scores = sess.run(None, {{"input": inp}})
        label = classes[int(label_idx[0])]
        print(label)          # e.g. "HUM:ind"
        print(dict(zip(classes, scores[0])))
        ```

        {prob_note}

        ## Labels ({n_cls} classes)

        {classes_md}

        ## Datasets

        Two datasets were built specifically for this project:

        - [{EAT_DATASET_URL}]({EAT_DATASET_URL}) — Expected Answer Type taxonomy,
          30K English questions, 53 fine-grained labels across 7 main categories
          (ABBR, BOOL, DESC, ENTY, HUM, LOC, NUM).
        - [{SENTENCE_TYPES_DATASET_URL}]({SENTENCE_TYPES_DATASET_URL}) — sentence-type
          taxonomy (question/statement/command/exclamation/request), 80K multilingual samples.

        ## Project

        Part of [little_questions]({LITTLE_QUESTIONS_URL}).
        """)


def _m2v_readme(variant_name: str, meta: dict) -> str:
    n_cls = len(meta.get("classes", []))
    classes_md = "\n".join(f"  - `{c}`" for c in (meta.get("classes") or []))
    m2v_model = meta.get("model_name", "minishlab/potion-base-8M")
    use_tfidf = meta.get("use_tfidf_fusion", False)
    fusion_note = " Embeddings are fused with TF-IDF word (1,2)-gram features." if use_tfidf else ""

    return dedent(f"""\
        ---
        language:
          - en
        license: apache-2.0
        tags:
          - question-classification
          - text-classification
          - model2vec
          - english
          - eat
        datasets:
          - TigreGotico/EAT
        ---

        # {variant_name}

        EAT question-type classifier — {n_cls} classes, Model2Vec backbone (`{m2v_model}`).{fusion_note}

        Trained on the [EAT dataset]({EAT_DATASET_URL}) (30K English questions).
        Requires `model2vec` and `scikit-learn` at inference.

        ## Usage

        ```python
        from train.classifiers import Model2VecClassifier

        clf = Model2VecClassifier.load("/path/to/{variant_name}")
        print(clf.predict(["Who invented the telephone?"]))
        ```

        ## Labels ({n_cls} classes)

        {classes_md}

        ## Datasets

        Two datasets were built specifically for this project:

        - [{EAT_DATASET_URL}]({EAT_DATASET_URL}) — Expected Answer Type taxonomy,
          30K English questions, 53 fine-grained labels across 7 main categories.
        - [{SENTENCE_TYPES_DATASET_URL}]({SENTENCE_TYPES_DATASET_URL}) — sentence-type
          taxonomy, 80K multilingual samples.

        ## Project

        Part of [little_questions]({LITTLE_QUESTIONS_URL}).
        """)


# ---------------------------------------------------------------------------
# Push helpers
# ---------------------------------------------------------------------------

def _repo_id(name: str) -> str:
    return f"{HF_ORG}/{name}"


def push_onnx_model(onnx_path: Path, dry_run: bool = False) -> None:
    from huggingface_hub import HfApi
    import onnxruntime as rt

    stem = onnx_path.stem  # e.g. eat53_svm_cal_EN_0.9.0
    # derive repo name: eat53_svm_cal_en  (lowercase lang, drop version)
    parts = stem.split("_")
    # format: eat{N}_{type(s)}_{LANG}_{VERSION}
    # repo name: eat{N}_{type(s)}_en
    repo_name = "_".join(parts[:-2] + [parts[-2].lower()])  # drop version, lowercase lang
    repo_id = _repo_id(repo_name)

    # Extract metadata from ONNX
    sess = rt.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    classes: list[str] = json.loads(meta.get("classes", "[]"))
    calibrated = meta.get("calibrated") == "true"

    # Derive n_cls and model_type from stem
    n_cls = int(parts[0].replace("eat", ""))
    model_type = "_".join(parts[1:-2])

    readme = _onnx_readme(stem, classes, n_cls, model_type, calibrated)

    print(f"{'[DRY RUN] ' if dry_run else ''}Pushing {onnx_path.name} → {repo_id}")
    if dry_run:
        return

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=str(onnx_path), path_in_repo=onnx_path.name,
                    repo_id=repo_id, repo_type="model")
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md",
                    repo_id=repo_id, repo_type="model")
    print(f"  → https://huggingface.co/{repo_id}")


def push_m2v_model(m2v_dir: Path, dry_run: bool = False) -> None:
    from huggingface_hub import HfApi

    variant_name = m2v_dir.name
    repo_id = _repo_id(variant_name)

    meta_path = m2v_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    readme = _m2v_readme(variant_name, meta)

    print(f"{'[DRY RUN] ' if dry_run else ''}Pushing {variant_name}/ → {repo_id}")
    if dry_run:
        return

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(m2v_dir), repo_id=repo_id, repo_type="model")
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md",
                    repo_id=repo_id, repo_type="model")
    print(f"  → https://huggingface.co/{repo_id}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Push EAT models to HuggingFace Hub")
    parser.add_argument("--type", choices=["onnx", "m2v", "all"], default="all")
    parser.add_argument("--model", help="Push a single model by stem name or directory name")
    parser.add_argument("--dry-run", action="store_true", help="Print repos without pushing")
    args = parser.parse_args()

    push_onnx = args.type in ("onnx", "all")
    push_m2v = args.type in ("m2v", "all")

    if push_onnx:
        onnx_files = sorted(ONNX_MODEL_DIR.glob("*.onnx")) if ONNX_MODEL_DIR.exists() else []
        for f in onnx_files:
            if args.model and f.stem != args.model:
                continue
            try:
                push_onnx_model(f, dry_run=args.dry_run)
            except Exception as exc:
                LOG.error("Failed to push %s: %s", f.name, exc)

    if push_m2v:
        m2v_dirs = sorted(M2V_MODEL_DIR.iterdir()) if M2V_MODEL_DIR.exists() else []
        for d in m2v_dirs:
            if not d.is_dir():
                continue
            if args.model and d.name != args.model:
                continue
            try:
                push_m2v_model(d, dry_run=args.dry_run)
            except Exception as exc:
                LOG.error("Failed to push %s: %s", d.name, exc)


if __name__ == "__main__":
    main()
