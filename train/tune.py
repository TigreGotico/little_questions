"""Optuna-based hyperparameter optimisation for COSC classifiers.

Each Optuna trial is logged as a nested MLflow run so you can browse
the full search history in the MLflow UI.

Usage::

    # Tune SVM for EN, 6-class (50 trials)
    python -m train.tune --lang en --classes 6 --trials 50

    # Tune model2vec variant for EN
    python -m train.tune --lang en --classes 6 --model m2v --trials 30

    # Tune all languages
    python -m train.tune --all-langs --classes 6 --trials 30
"""

from __future__ import annotations

import argparse
import logging
import os
from os.path import dirname, join
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import f1_score

import train.mlflow_config as mlflow_config
from train.utils import load_data

LOG = logging.getLogger(__name__)

DATA_DIR = join(dirname(__file__), "clean_data")

LANG_CONFIG: Dict[str, Dict[str, str]] = {
    "en": {"dataset": "raw_questions_EN_balanced_0.8.0.txt"},
    "es": {"dataset": "raw_questions_ES_googtx0.7.0a1.txt"},
    "ca": {"dataset": "raw_questions_CA_apertiumtx0.7.0a1.txt"},
    "pt": {"dataset": "raw_questions_PT_googtx0.7.0a1.txt"},
    "fr": {"dataset": "raw_questions_FR_googtx0.7.0a1.txt"},
    "de": {"dataset": "raw_questions_DE_0.8.0.txt"},
    "it": {"dataset": "raw_questions_IT_0.8.0.txt"},
    "nl": {"dataset": "raw_questions_NL_0.8.0.txt"},
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load(lang: str, classes: int) -> Tuple[List[str], List[str]]:
    config = LANG_CONFIG[lang]
    data_path = join(DATA_DIR, config["dataset"])
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    return load_data(data_path, classes)


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def _build_tfidf_features(
    trial,
    x_train: List[str],
    x_val: List[str],
    lang: str,
    use_linguistic: bool,
):
    """Build TF-IDF (± linguistic) feature matrices for one trial."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.sparse import hstack

    # TF-IDF word n-gram hyperparams
    ngram_max = trial.suggest_int("ngram_max", 1, 3)
    min_df = trial.suggest_int("min_df", 1, 3)
    max_df = trial.suggest_float("max_df", 0.3, 0.95)
    use_char = trial.suggest_categorical("use_char", [True, False])

    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, ngram_max),
        min_df=min_df, max_df=max_df, sublinear_tf=True,
    )
    X_tr = word_vec.fit_transform(x_train)
    X_v  = word_vec.transform(x_val)

    if use_char:
        char_ngram_min = trial.suggest_int("char_ngram_min", 2, 3)
        char_ngram_max = trial.suggest_int("char_ngram_max", 3, 5)
        if char_ngram_min > char_ngram_max:
            char_ngram_max = char_ngram_min
        char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(char_ngram_min, char_ngram_max),
            min_df=min_df, max_df=max_df,
        )
        X_tr = hstack([X_tr, char_vec.fit_transform(x_train)])
        X_v  = hstack([X_v,  char_vec.transform(x_val)])

    if use_linguistic:
        from train.features import LinguisticFeaturesTransformer
        ling = LinguisticFeaturesTransformer(sparse=True)
        X_tr = hstack([X_tr, ling.fit_transform(x_train)])
        X_v  = hstack([X_v,  ling.transform(x_val)])

    return X_tr, X_v


def _build_m2v_features(trial, x_train, x_val, use_tfidf: bool):
    """Build model2vec (± TF-IDF) feature matrices for one trial."""
    from model2vec import StaticModel
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer

    model_name = trial.suggest_categorical("m2v_model", [
        "minishlab/potion-base-2M",
        "minishlab/potion-base-8M",
        "minishlab/potion-base-32M",
        "minishlab/potion-multilingual-128M",
    ])
    m2v = StaticModel.from_pretrained(model_name)
    X_tr = m2v.encode(x_train)
    X_v  = m2v.encode(x_val)

    if use_tfidf:
        ngram_max = trial.suggest_int("tfidf_ngram_max", 1, 2)
        tfidf = TfidfVectorizer(
            analyzer="word", ngram_range=(1, ngram_max),
            min_df=1, max_df=0.5, sublinear_tf=True,
        )
        X_tr = hstack([tfidf.fit_transform(x_train), X_tr])
        X_v  = hstack([tfidf.transform(x_val),       X_v])

    return X_tr, X_v


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------

def _svm_objective(
    trial,
    x_train: List[str],
    y_train: List[str],
    x_val:   List[str],
    y_val:   List[str],
    lang: str,
    classes: int,
    use_linguistic: bool,
    parent_run_id: str,
) -> float:
    import mlflow
    from sklearn.svm import LinearSVC

    C = trial.suggest_float("C", 1e-3, 100.0, log=True)
    X_tr, X_v = _build_tfidf_features(trial, x_train, x_val, lang, use_linguistic)
    clf = LinearSVC(C=C, max_iter=2000)
    clf.fit(X_tr, y_train)
    preds = clf.predict(X_v)
    macro_f1 = f1_score(y_val, preds, average="macro", zero_division=0)

    # Log trial as nested MLflow run
    with mlflow.start_run(
        run_name=f"trial-{trial.number}",
        nested=True,
        tags={"parent_run_id": parent_run_id},
    ):
        mlflow.log_params({**trial.params, "lang": lang, "n_classes": classes,
                           "model_type": "tfidf-svm"})
        mlflow.log_metric("macro_f1", macro_f1)
        mlflow.log_metric("trial", trial.number)

    return macro_f1


def _m2v_objective(
    trial,
    x_train: List[str],
    y_train: List[str],
    x_val:   List[str],
    y_val:   List[str],
    lang: str,
    classes: int,
    parent_run_id: str,
) -> float:
    import mlflow
    from sklearn.svm import LinearSVC

    use_tfidf = trial.suggest_categorical("use_tfidf_fusion", [True, False])
    C = trial.suggest_float("C", 1e-3, 100.0, log=True)

    X_tr, X_v = _build_m2v_features(trial, x_train, x_val, use_tfidf)
    clf = LinearSVC(C=C, max_iter=2000)
    clf.fit(X_tr, y_train)
    preds = clf.predict(X_v)
    macro_f1 = f1_score(y_val, preds, average="macro", zero_division=0)

    with mlflow.start_run(
        run_name=f"trial-{trial.number}",
        nested=True,
        tags={"parent_run_id": parent_run_id},
    ):
        mlflow.log_params({**trial.params, "lang": lang, "n_classes": classes,
                           "model_type": "m2v"})
        mlflow.log_metric("macro_f1", macro_f1)
        mlflow.log_metric("trial", trial.number)

    return macro_f1


# ---------------------------------------------------------------------------
# SGD learning-curve trainer (true per-epoch metrics → MLflow)
# ---------------------------------------------------------------------------

def train_sgd_epochs(
    x_train: List[str],
    y_train: List[str],
    x_val:   List[str],
    y_val:   List[str],
    lang: str,
    classes: int,
    n_epochs: int = 20,
    use_linguistic: bool = False,
) -> Dict[str, Any]:
    """Train SGDClassifier with per-epoch metrics logged to MLflow.

    Logs train accuracy, val accuracy, train macro-F1, and val macro-F1 at
    every epoch.  Also logs a learning-curve plot as an artefact.

    Returns:
        dict with ``best_epoch``, ``best_val_f1``, ``model``.
    """
    import mlflow
    from sklearn.linear_model import SGDClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score
    from scipy.sparse import hstack, issparse
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Build features once
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=1, max_df=0.4, sublinear_tf=True,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 4), min_df=1, max_df=0.4,
    )
    X_tr = hstack([word_vec.fit_transform(x_train), char_vec.fit_transform(x_train)])
    X_v  = hstack([word_vec.transform(x_val),       char_vec.transform(x_val)])

    if use_linguistic:
        from train.features import LinguisticFeaturesTransformer
        ling = LinguisticFeaturesTransformer(sparse=True)
        X_tr = hstack([X_tr, ling.fit_transform(x_train)])
        X_v  = hstack([X_v,  ling.transform(x_val)])

    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_train)
    y_v_enc  = le.transform(y_val)
    classes_arr = np.arange(len(le.classes_))

    clf = SGDClassifier(loss="hinge", penalty="l2", alpha=1e-4,
                        random_state=42, tol=None, max_iter=1)

    history: Dict[str, List[float]] = {
        "train_acc": [], "val_acc": [], "train_f1": [], "val_f1": [],
    }
    best_val_f1, best_epoch = 0.0, 0

    run_name = f"sgd-epochs-{lang}-{classes}c"
    mlflow.set_experiment(mlflow_config.EXPERIMENT_COSC)
    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.log_params({
            "lang": lang, "n_classes": classes, "model_type": "sgd-tfidf",
            "n_epochs": n_epochs, "use_linguistic": use_linguistic,
        })

        for epoch in range(1, n_epochs + 1):
            # Shuffle train set each epoch
            idx = np.random.permutation(X_tr.shape[0])
            clf.partial_fit(X_tr[idx], y_tr_enc[idx], classes=classes_arr)

            tr_pred = clf.predict(X_tr)
            v_pred  = clf.predict(X_v)

            tr_acc = accuracy_score(y_tr_enc, tr_pred)
            v_acc  = accuracy_score(y_v_enc,  v_pred)
            tr_f1  = f1_score(y_tr_enc, tr_pred, average="macro", zero_division=0)
            v_f1   = f1_score(y_v_enc,  v_pred,  average="macro", zero_division=0)

            history["train_acc"].append(tr_acc)
            history["val_acc"].append(v_acc)
            history["train_f1"].append(tr_f1)
            history["val_f1"].append(v_f1)

            mlflow.log_metrics({
                "train_accuracy": tr_acc,
                "val_accuracy":   v_acc,
                "train_macro_f1": tr_f1,
                "val_macro_f1":   v_f1,
            }, step=epoch)

            if v_f1 > best_val_f1:
                best_val_f1, best_epoch = v_f1, epoch
            LOG.info("epoch %02d  train_f1=%.4f  val_f1=%.4f", epoch, tr_f1, v_f1)

        # Learning-curve plot
        epochs = list(range(1, n_epochs + 1))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(epochs, history["train_acc"], label="train")
        ax1.plot(epochs, history["val_acc"],   label="val")
        ax1.set_title(f"Accuracy per epoch — {lang.upper()} {classes}-class")
        ax1.set_xlabel("Epoch")
        ax1.legend()

        ax2.plot(epochs, history["train_f1"], label="train")
        ax2.plot(epochs, history["val_f1"],   label="val")
        ax2.set_title(f"Macro F1 per epoch — {lang.upper()} {classes}-class")
        ax2.set_xlabel("Epoch")
        ax2.legend()

        fig.tight_layout()
        plot_path = f"/tmp/sgd_learning_curve_{lang}_{classes}c.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(plot_path, artifact_path="plots")

        mlflow.log_metrics({
            "best_val_macro_f1": best_val_f1,
            "best_epoch": best_epoch,
        })

    return {"best_epoch": best_epoch, "best_val_f1": best_val_f1, "history": history}


# ---------------------------------------------------------------------------
# Main tuning entrypoint
# ---------------------------------------------------------------------------

def tune(
    lang: str,
    classes: int,
    model: str = "svm",
    n_trials: int = 50,
    use_linguistic: bool = True,
) -> Dict[str, Any]:
    """Run Optuna HPO for *lang* and log all trials to MLflow.

    Args:
        lang: Language code.
        classes: 6 or 52.
        model: ``"svm"`` or ``"m2v"``.
        n_trials: Number of Optuna trials.
        use_linguistic: Include linguistic features (SVM only).

    Returns:
        dict with ``best_params``, ``best_value``.
    """
    import mlflow
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    x, y = _load(lang, classes)
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.15, stratify=y, random_state=42,
    )

    experiment = mlflow_config.EXPERIMENT_COMPARE
    run_name = f"optuna-{model}-{lang}-{classes}c"
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name) as parent_run:
        mlflow.log_params({
            "lang": lang, "n_classes": classes, "model_type": model,
            "n_trials": n_trials, "optimizer": "optuna",
            "use_linguistic": use_linguistic,
        })

        if model == "svm":
            objective = lambda t: _svm_objective(
                t, x_train, y_train, x_val, y_val,
                lang, classes, use_linguistic, parent_run.info.run_id,
            )
        else:
            objective = lambda t: _m2v_objective(
                t, x_train, y_train, x_val, y_val,
                lang, classes, parent_run.info.run_id,
            )

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best = study.best_trial
        print(f"\n{'=' * 60}")
        print(f"Best trial #{best.number}  macro_f1={best.value:.4f}")
        print(f"Params: {best.params}")

        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
        mlflow.log_metric("best_macro_f1", best.value)
        mlflow.log_metric("best_trial_number", best.number)

        # Optuna visualizations
        try:
            import matplotlib
            matplotlib.use("Agg")
            from optuna.visualization.matplotlib import (
                plot_optimization_history,
                plot_param_importances,
            )
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            plot_optimization_history(study, ax=axes[0])
            axes[0].set_title(f"Optimization history — {lang.upper()} {model} {classes}c")
            try:
                plot_param_importances(study, ax=axes[1])
                axes[1].set_title("Param importances")
            except Exception:
                axes[1].set_visible(False)
            fig.tight_layout()
            plot_path = f"/tmp/optuna_{model}_{lang}_{classes}c.png"
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(plot_path, artifact_path="plots")
        except Exception as exc:
            LOG.warning("Could not generate Optuna plots: %s", exc)

    return {"best_params": best.params, "best_value": best.value}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Optuna HPO for COSC classifiers with MLflow logging"
    )
    parser.add_argument("--lang", default="en")
    parser.add_argument("--all-langs", action="store_true")
    parser.add_argument("--classes", type=int, default=6, choices=[6, 52])
    parser.add_argument("--model", default="svm", choices=["svm", "m2v"])
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--no-linguistic", action="store_true")
    parser.add_argument("--epochs", action="store_true",
                        help="Run SGD per-epoch training instead of Optuna")
    parser.add_argument("--n-epochs", type=int, default=20)
    args = parser.parse_args()

    mlflow_config.setup()

    langs = list(LANG_CONFIG) if args.all_langs else [args.lang]

    for lang in langs:
        if args.epochs:
            x, y = _load(lang, args.classes)
            x_train, x_val, y_train, y_val = train_test_split(
                x, y, test_size=0.15, stratify=y, random_state=42,
            )
            train_sgd_epochs(x_train, y_train, x_val, y_val,
                             lang, args.classes, n_epochs=args.n_epochs,
                             use_linguistic=not args.no_linguistic)
        else:
            tune(lang, args.classes, model=args.model,
                 n_trials=args.trials, use_linguistic=not args.no_linguistic)


if __name__ == "__main__":
    main()
