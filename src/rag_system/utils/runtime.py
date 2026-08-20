"""Device selection, seeding and small runtime helpers."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

LOGGER_NAME = "rag_system"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


def get_logger() -> logging.Logger:
    return setup_logging()


def resolve_device(requested: Optional[str] = None) -> str:
    """Pick a torch device.

    Order is MPS > CUDA > CPU because this project was developed on Apple
    silicon; an explicit ``requested`` value always wins so benchmarks can pin
    the device.
    """
    import torch

    if requested:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch (all devices)."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch is optional for pure-data unit tests
        pass


@contextmanager
def timer(label: str = "", logger: Optional[logging.Logger] = None) -> Iterator[dict]:
    """Time a block; the yielded dict is filled with ``seconds`` on exit."""
    result: dict = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start
        if label and logger:
            logger.info("%s took %.2fs", label, result["seconds"])


def rss_memory_mb() -> float:
    """Current resident set size of this process, in MB.

    This is *current* RSS, not a high-water mark: psutil's ``memory_info().rss``
    reports the value at call time, and it can fall after tensors are freed.
    Anything labelled "peak" here would be wrong.
    """
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024**2)
    except Exception:
        return float("nan")


def write_json(path: Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_default))
    return path


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def _default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "to_dict"):
        return o.to_dict()
    raise TypeError(f"not JSON serialisable: {type(o)}")
