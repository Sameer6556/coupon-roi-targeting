"""Thin MLflow wrapper. If MLflow isn't installed or tracking is off, everything still runs.

Runs are stored in a local SQLite file (mlflow.db) with artifacts in mlruns/.
Browse them with:  mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterator

from . import config

log = logging.getLogger(__name__)
EXPERIMENT = "coupon-roi-targeting"


@contextlib.contextmanager
def run(name: str, enabled: bool = True, tags: dict | None = None) -> Iterator[object | None]:
    if not enabled:
        yield None
        return
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    try:
        import mlflow
    except ImportError:
        log.info("mlflow not installed; skipping tracking for %s", name)
        yield None
        return
    for noisy in ("mlflow", "alembic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    mlflow.set_tracking_uri(f"sqlite:///{config.ROOT / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        config.MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(EXPERIMENT, artifact_location=config.MLFLOW_DIR.as_uri())
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=name, tags=tags or {}):
        yield mlflow
