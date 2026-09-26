"""MLflow configuration for little_questions training runs.

Reads connection settings from environment variables so credentials are never
hard-coded in source files.  Set these before running any train script:

    export MLFLOW_TRACKING_URI=https://mlflow.tigregotico.pt
    export MLFLOW_TRACKING_USERNAME=admin
    export MLFLOW_TRACKING_PASSWORD=<token>

Or pass them directly in the shell:

    MLFLOW_TRACKING_URI=... python -m train.train_all
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Optional

LOG = logging.getLogger(__name__)

# Default experiment names
EXPERIMENT_COSC = "little-questions-cosc"
EXPERIMENT_SENTENCE_TYPE = "little-questions-sentence-type"
EXPERIMENT_COMPARE = "little-questions-compare"


def setup(
    tracking_uri: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> bool:
    """Configure MLflow tracking from arguments or environment variables.

    Returns True if a remote tracking URI was configured, False if falling
    back to local file-based tracking.
    """
    import mlflow

    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        LOG.info("MLFLOW_TRACKING_URI not set — using local ./mlruns")
        return False

    if username or os.environ.get("MLFLOW_TRACKING_USERNAME"):
        os.environ.setdefault(
            "MLFLOW_TRACKING_USERNAME",
            username or os.environ["MLFLOW_TRACKING_USERNAME"],
        )
    if password or os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        os.environ.setdefault(
            "MLFLOW_TRACKING_PASSWORD",
            password or os.environ["MLFLOW_TRACKING_PASSWORD"],
        )

    mlflow.set_tracking_uri(uri)
    LOG.info("MLflow tracking URI: %s", uri)
    return True


@contextmanager
def run(
    experiment: str,
    run_name: str,
    params: Dict[str, Any],
    tags: Optional[Dict[str, str]] = None,
):
    """Context manager that wraps an mlflow run.

    Yields the active run.  Logs *params* on entry; caller logs metrics
    and artefacts inside the ``with`` block.

    Example::

        with mlflow_config.run("little-questions-cosc", "svm-en", {"lang": "en"}) as active_run:
            mlflow.log_metric("accuracy", 0.92)
    """
    import mlflow

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name, tags=tags or {}) as active_run:
        mlflow.log_params(params)
        yield active_run


def log_classification_metrics(
    y_true: list,
    y_pred: list,
    prefix: str = "",
) -> Dict[str, float]:
    """Log accuracy, macro F1, and weighted F1 to the active MLflow run.

    Returns the dict of logged metrics.
    """
    import mlflow
    from sklearn.metrics import accuracy_score, f1_score

    sep = f"{prefix}_" if prefix else ""
    metrics = {
        f"{sep}accuracy": accuracy_score(y_true, y_pred),
        f"{sep}macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        f"{sep}weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    mlflow.log_metrics(metrics)
    return metrics
