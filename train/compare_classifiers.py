#!/usr/bin/env python3
"""Benchmark TF-IDF SVM vs model2vec Potion vs heuristic baseline.

Every run is logged to MLflow.  Set connection env vars before running:

    export MLFLOW_TRACKING_URI=https://mlflow.tigregotico.pt
    export MLFLOW_TRACKING_USERNAME=admin
    export MLFLOW_TRACKING_PASSWORD=<token>

Usage::

    # Compare all models on EN, 6-class
    python -m train.compare_classifiers

    # 52-class, specific language
    python -m train.compare_classifiers --lang es --classes 52

    # All languages with enhanced features, produce plots
    python -m train.compare_classifiers --all-langs --classes 6 --enhanced --save-plots
"""

from __future__ import annotations

import argparse
import logging
import os
from os.path import dirname, join
from pathlib import Path
from typing import Dict, List

from sklearn.model_selection import train_test_split

import train.mlflow_config as mlflow_config
from train.classifiers import LinearSVCClassifier, Model2VecClassifier
from train.baselines import HeuristicScorer, PunctuationScorer
from train.metrics import EvalResult, evaluate, compare
from train.utils import load_data

LOG = logging.getLogger(__name__)

DATA_DIR = join(dirname(__file__), "clean_data")
REPORTS_DIR = join(dirname(__file__), "reports")

LANG_CONFIG: Dict[str, Dict[str, str]] = {
    "en": {"dataset": "raw_questions_EN_balanced_0.8.0.txt", "suffix": "EN"},
    "es": {"dataset": "raw_questions_ES_balanced_0.8.0.txt",  "suffix": "ES"},
    "ca": {"dataset": "raw_questions_CA_apertiumtx0.7.0a1.txt", "suffix": "CA"},
    "pt": {"dataset": "raw_questions_PT_balanced_0.8.0.txt",  "suffix": "PT"},
    "fr": {"dataset": "raw_questions_FR_balanced_0.8.0.txt",  "suffix": "FR"},
    "de": {"dataset": "raw_questions_DE_balanced_0.8.0.txt",  "suffix": "DE"},
    "it": {"dataset": "raw_questions_IT_balanced_0.8.0.txt",  "suffix": "IT"},
    "nl": {"dataset": "raw_questions_NL_balanced_0.8.0.txt",  "suffix": "NL"},
}

# Potion models to benchmark.  EN-only models are skipped for non-EN langs.
POTION_MODELS = [
    ("minishlab/potion-base-2M",         "en_only"),
    ("minishlab/potion-base-8M",         "en_only"),
    ("minishlab/potion-base-32M",        "en_only"),
    ("minishlab/potion-multilingual-128M", "multilingual"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _SklearnScorerAdapter:
    """Wrap a sklearn-style classifier so it matches the Scorer protocol."""

    def __init__(self, name: str, clf) -> None:
        self.name = name
        self._clf = clf

    def predict(self, text: str) -> str:
        return self._clf.predict([text])[0]


def _run_baselines(
    lang: str,
    classes: int,
    x_test: List[str],
    y_test: List[str],
) -> List[EvalResult]:
    """Evaluate rule-based baselines and log each to MLflow."""
    results = []

    scorers = [PunctuationScorer()]
    if lang == "en":
        scorers.append(HeuristicScorer())

    for scorer in scorers:
        name = getattr(scorer, "name", type(scorer).__name__)
        params = {"lang": lang, "n_classes": classes, "model_type": "heuristic",
                  "scorer": name}
        with mlflow_config.run(mlflow_config.EXPERIMENT_COMPARE,
                               f"{name}-{lang}-{classes}c", params):
            result = evaluate(scorer, x_test, y_test, lang=lang,
                              dataset=f"test-split-{lang}-{classes}c")
            mlflow_config.log_classification_metrics(y_test,
                                                     [scorer.predict(t) for t in x_test])
            _log_report_artifact(result, f"{name}_{lang}_{classes}c")
        results.append(result)
        print(f"  {name}: accuracy={result.accuracy:.4f}  macro_f1={result.macro_f1:.4f}")

    return results


def _load_split(lang: str, classes: int):
    """Return (x_train, x_test, y_train, y_test) for *lang*."""
    config = LANG_CONFIG[lang]
    data_path = join(DATA_DIR, config["dataset"])
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    x, y = load_data(data_path, classes)
    return train_test_split(x, y, test_size=0.15, stratify=y, random_state=42)


# ---------------------------------------------------------------------------
# Single-classifier benchmark
# ---------------------------------------------------------------------------

def _run_enhanced_svm(
    lang: str,
    classes: int,
    x_train, x_test, y_train, y_test,
) -> EvalResult:
    """Train TF-IDF + char n-gram + linguistic features LinearSVC."""
    import mlflow
    from sklearn.svm import LinearSVC
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.sparse import hstack
    from train.features import LinguisticFeaturesTransformer

    params = {
        "lang": lang, "n_classes": classes,
        "model_type": "tfidf+char+linguistic-svm",
        "vectorizer": "tfidf-word12+char34+linguistic",
        "classifier": "LinearSVC",
    }
    run_name = f"enhanced-svm-{lang}-{classes}c"

    with mlflow_config.run(mlflow_config.EXPERIMENT_COMPARE, run_name, params):
        word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                   min_df=1, max_df=0.4, sublinear_tf=True)
        char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4),
                                   min_df=1, max_df=0.4)
        ling = LinguisticFeaturesTransformer(sparse=True)

        X_tr = hstack([word_vec.fit_transform(x_train),
                       char_vec.fit_transform(x_train),
                       ling.fit_transform(x_train)])
        X_te = hstack([word_vec.transform(x_test),
                       char_vec.transform(x_test),
                       ling.transform(x_test)])

        clf = LinearSVC(C=1.0, max_iter=2000)
        clf.fit(X_tr, y_train)

        class _Adapter:
            name = run_name
            def predict(_, text: str) -> str:
                from scipy.sparse import hstack as _h
                feat = _h([word_vec.transform([text]),
                           char_vec.transform([text]),
                           ling.transform([text])])
                return clf.predict(feat)[0]

        result = evaluate(_Adapter(), x_test, y_test, lang=lang,
                          dataset=f"test-split-{lang}-{classes}c")
        mlflow_config.log_classification_metrics(y_test, clf.predict(X_te))
        _log_report_artifact(result, run_name)

    return result


def _run_svm(
    lang: str,
    classes: int,
    x_train, x_test, y_train, y_test,
) -> EvalResult:
    """Train TF-IDF LinearSVC and log to MLflow."""
    import mlflow
    from xdg import BaseDirectory as XDG

    clf = LinearSVCClassifier(lang)
    params = {
        "lang": lang, "n_classes": classes, "model_type": "tfidf-svm",
        "vectorizer": "tfidf-word12", "classifier": "LinearSVC",
    }
    with mlflow_config.run(mlflow_config.EXPERIMENT_COMPARE, f"svm-{lang}-{classes}c", params):
        clf.train(x_train, y_train)
        adapter = _SklearnScorerAdapter("tfidf-svm", clf)
        result = evaluate(adapter, x_test, y_test, lang=lang,
                          dataset=f"test-split-{lang}-{classes}c")
        mlflow_config.log_classification_metrics(y_test, clf.predict(x_test))
        _log_report_artifact(result, f"svm_{lang}_{classes}c")

        # Export and log ONNX
        model_dir = XDG.save_data_path("little_questions")
        onnx_path = join(model_dir, f"questions{classes}_svm_{lang.upper()}_0.8.0.onnx")
        try:
            clf.save_onnx(onnx_path)
            mlflow.log_artifact(onnx_path, artifact_path="onnx")
            LOG.info("Logged ONNX: %s", onnx_path)
        except Exception as exc:
            LOG.warning("ONNX export skipped: %s", exc)

    return result


def _run_m2v(
    lang: str,
    classes: int,
    model_name: str,
    use_tfidf_fusion: bool,
    x_train, x_test, y_train, y_test,
) -> EvalResult | None:
    """Train Model2Vec classifier and log to MLflow."""
    short = model_name.split("/")[-1]
    fusion_tag = "+tfidf" if use_tfidf_fusion else ""
    run_name = f"m2v-{short}{fusion_tag}-{lang}-{classes}c"

    params = {
        "lang": lang, "n_classes": classes,
        "model_type": f"m2v{fusion_tag}",
        "m2v_model": model_name,
        "tfidf_fusion": use_tfidf_fusion,
        "classifier": "LinearSVC",
    }

    try:
        clf = Model2VecClassifier(model_name, use_tfidf_fusion=use_tfidf_fusion)
        with mlflow_config.run(mlflow_config.EXPERIMENT_COMPARE, run_name, params) as active_run:
            import mlflow
            clf.train(x_train, y_train)
            adapter = _SklearnScorerAdapter(run_name, clf)
            result = evaluate(adapter, x_test, y_test, lang=lang,
                              dataset=f"test-split-{lang}-{classes}c")
            mlflow_config.log_classification_metrics(y_test, clf.predict(x_test))
            _log_report_artifact(result, run_name.replace("/", "_"))

            # Save model artefacts and log to MLflow
            model_out = join(REPORTS_DIR, "m2v_models", run_name.replace("/", "_"))
            clf.save(model_out)
            mlflow.log_artifacts(model_out, artifact_path="model2vec")

        return result
    except Exception as exc:
        LOG.warning("Skipped %s: %s", run_name, exc)
        return None


def _log_report_artifact(result: EvalResult, stem: str) -> None:
    """Save classification report as a text artefact and upload to MLflow."""
    import mlflow

    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = join(REPORTS_DIR, f"{stem}.txt")
    Path(report_path).write_text(str(result), encoding="utf-8")
    mlflow.log_artifact(report_path)


# ---------------------------------------------------------------------------
# Per-language comparison
# ---------------------------------------------------------------------------

def compare_lang(
    lang: str,
    classes: int = 6,
    save_plots: bool = False,
    show_plots: bool = True,
    run_enhanced: bool = False,
) -> List[EvalResult]:
    """Run all classifiers for *lang* and return results."""
    print(f"\n{'=' * 70}")
    print(f"BENCHMARK  lang={lang.upper()}  classes={classes}")
    print("=" * 70)

    x_train, x_test, y_train, y_test = _load_split(lang, classes)
    print(f"Train={len(x_train)}  Test={len(x_test)}  Labels={len(set(y_train))}")

    results: List[EvalResult] = []

    # 0. Heuristic baselines (no training — evaluate on full test set)
    print("\n[0] Heuristic baselines ...")
    results.extend(_run_baselines(lang, classes, x_test, y_test))

    # 1. TF-IDF SVM
    print("\n[1] TF-IDF LinearSVC ...")
    results.append(_run_svm(lang, classes, x_train, x_test, y_train, y_test))

    # 2. Enhanced SVM (TF-IDF + char + linguistic features)
    if run_enhanced:
        print("\n[2] Enhanced SVM (TF-IDF + char + linguistic) ...")
        results.append(_run_enhanced_svm(lang, classes, x_train, x_test, y_train, y_test))

    # 3. Potion / model2vec variants
    for idx, (model_name, scope) in enumerate(POTION_MODELS, start=3):
        is_multilingual = scope == "multilingual"
        if lang != "en" and not is_multilingual:
            LOG.debug("Skipping EN-only model %s for lang=%s", model_name, lang)
            continue
        short = model_name.split("/")[-1]
        print(f"\n[{idx}] {short} (alone) ...")
        r = _run_m2v(lang, classes, model_name, False, x_train, x_test, y_train, y_test)
        if r:
            results.append(r)
        print(f"\n[{idx}] {short} + TF-IDF fusion ...")
        r = _run_m2v(lang, classes, model_name, True, x_train, x_test, y_train, y_test)
        if r:
            results.append(r)

    # 4. Print comparison table
    print()
    compare(
        results,
        save_path=join(REPORTS_DIR, f"compare_{lang}_{classes}c.png") if save_plots else None,
    )
    if show_plots and not save_plots:
        compare(results)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Benchmark TF-IDF SVM vs Potion model2vec for COSC classification"
    )
    parser.add_argument("--lang", default="en", help="Language code (default: en)")
    parser.add_argument(
        "--all-langs", action="store_true", help="Benchmark all 8 languages"
    )
    parser.add_argument(
        "--classes", type=int, default=6, choices=[6, 52],
        help="Number of COSC classes (default: 6)"
    )
    parser.add_argument("--plot", action="store_true", help="Show plots interactively")
    parser.add_argument("--save-plots", action="store_true", help="Save plots to reports/")
    parser.add_argument("--enhanced", action="store_true",
                        help="Also run enhanced SVM (TF-IDF + char + linguistic features)")
    args = parser.parse_args()

    mlflow_config.setup()

    langs = list(LANG_CONFIG) if args.all_langs else [args.lang]

    all_results: Dict[str, List[EvalResult]] = {}
    for lang in langs:
        try:
            all_results[lang] = compare_lang(
                lang,
                args.classes,
                save_plots=args.save_plots,
                show_plots=args.plot,
                run_enhanced=args.enhanced,
            )
        except FileNotFoundError as exc:
            LOG.warning("Skipping %s: %s", lang, exc)

    # Global summary across all languages
    if len(all_results) > 1:
        print(f"\n{'=' * 70}")
        print("GLOBAL SUMMARY")
        print("=" * 70)
        for lang, results in all_results.items():
            best = max(results, key=lambda r: r.macro_f1)
            print(f"  {lang.upper():<4}  best={best.scorer_name:<35}  macro_f1={best.macro_f1:.4f}")


if __name__ == "__main__":
    main()
