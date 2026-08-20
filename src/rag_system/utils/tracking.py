"""Thin MLflow wrapper.

MLflow is used with a local SQLite backend (``sqlite:///mlflow.db``) so the
project needs no tracking server and stays reproducible offline.  The plain
file store is deliberately not used: MLflow 3.x raises on it ("filesystem
tracking backend is in maintenance mode"), so SQLite is the supported local
equivalent.  Tracking degrades to a no-op if MLflow
is missing or misconfigured -- an experiment must never fail because its logger
did.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from .runtime import get_logger

LOGGER = get_logger()


class NullRun:
    def log_params(self, params: Dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None: ...
    def log_artifact(self, path: Path) -> None: ...
    def set_tags(self, tags: Dict[str, Any]) -> None: ...


class MlflowRun(NullRun):
    def __init__(self, mlflow_module):
        self._mlflow = mlflow_module

    def log_params(self, params: Dict[str, Any]) -> None:
        # MLflow rejects params over 500 chars and non-scalars.
        clean = {k: (str(v)[:480] if v is not None else "none") for k, v in params.items()}
        self._mlflow.log_params(clean)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        numeric = {
            k.replace("@", "_at_"): float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        if numeric:
            self._mlflow.log_metrics(numeric, step=step)

    def log_artifact(self, path: Path) -> None:
        if Path(path).exists():
            self._mlflow.log_artifact(str(path))

    def set_tags(self, tags: Dict[str, Any]) -> None:
        self._mlflow.set_tags({k: str(v) for k, v in tags.items()})


@contextmanager
def start_run(run_name: str, tracking_uri: str = "sqlite:///mlflow.db",
              experiment_name: str = "domain-rag-qa",
              enabled: bool = True) -> Iterator[NullRun]:
    """Open an MLflow run, or yield a no-op recorder if tracking is unavailable."""
    if not enabled:
        yield NullRun()
        return
    try:
        import mlflow
    except ImportError:
        LOGGER.warning("mlflow not installed; experiment tracking disabled")
        yield NullRun()
        return

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name):
            yield MlflowRun(mlflow)
    except Exception as exc:  # pragma: no cover - tracking must never break a run
        LOGGER.warning("mlflow disabled (%s: %s)", type(exc).__name__, exc)
        yield NullRun()
