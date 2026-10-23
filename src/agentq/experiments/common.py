"""Shared helpers for experiments: seed aggregation, confidence intervals, errors, result I/O."""

from __future__ import annotations

import json
import math
import platform
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy import stats

import agentq


def mean_ci(values: Sequence[float], level: float = 0.95) -> tuple[float, float]:
    """Mean and Student-t half-width (paper: 95% CIs for the journal-version experiments)."""
    x = np.asarray(values, dtype=float)
    if x.size < 2:
        return float(x.mean()), float("nan")
    half = stats.t.ppf(0.5 + level / 2, x.size - 1) * x.std(ddof=1) / math.sqrt(x.size)
    return float(x.mean()), float(half)


def mean_sd(values: Sequence[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (paper: conference-version experiments, 4 seeds)."""
    x = np.asarray(values, dtype=float)
    return float(x.mean()), float(x.std(ddof=1)) if x.size > 1 else float("nan")


def relative_error(model: float | None, reference: float) -> float | None:
    """``(model - reference) / reference`` (paper's error convention)."""
    if model is None or reference == 0:
        return None
    return model / reference - 1.0


def aggregate(runs: Sequence[Any], keys: Sequence[str], spread: str = "ci") -> dict[str, tuple[float, float]]:
    """Aggregate a list of result dataclasses over seeds: ``{key: (mean, sd or ci half-width)}``."""
    fn: Callable[[Sequence[float]], tuple[float, float]] = mean_ci if spread == "ci" else mean_sd
    return {k: fn([getattr(r, k) for r in runs]) for k in keys}


def environment_metadata() -> dict[str, str]:
    """Software/hardware metadata recorded with every result file."""
    return {
        "agentq": agentq.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
    }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.floating | np.integer):
        return obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if hasattr(obj, "to_dict"):
        return _jsonable(obj.to_dict())
    return obj


def save_result(path: Path, experiment: str, mode: str, params: dict[str, Any], data: Any, elapsed_s: float) -> Path:
    """Write ``data`` with provenance metadata to ``path`` (JSON)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": experiment,
        "mode": mode,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed_s, 2),
        "params": _jsonable(params),
        "environment": environment_metadata(),
        "provenance": "reproduced locally with agentq (not the paper's original output files)",
        "results": _jsonable(data),
    }
    path.write_text(json.dumps(payload, indent=1))
    return path


class Timer:
    """Context manager measuring wall time."""

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self.start
