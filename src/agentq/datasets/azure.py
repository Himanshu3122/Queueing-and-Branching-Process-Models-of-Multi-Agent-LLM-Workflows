"""Azure LLM inference trace 2023 (Patel et al., 2024) as a real task-arrival process.

File: ``AzureLLMInferenceTrace_conv.csv`` from ``github.com/Azure/AzurePublicDataset`` (CC-BY 4.0),
19,366 requests to a production conversation service over one hour on 2023-11-16 with columns
``TIMESTAMP, ContextTokens, GeneratedTokens``.

Used here only for its timestamps: each request is treated as one task arrival, and the trace is
rescaled in time to reach a target load. This keeps the real burst structure (interarrival
variability and long-range rate changes) while the workflow comes from AgentInstruct.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def _parse_ts(text: str) -> datetime:
    head, _, frac = text.strip().partition(".")
    return datetime.strptime(f"{head}.{(frac + '000000')[:6]}", "%Y-%m-%d %H:%M:%S.%f")


def load_azure_arrivals(path: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return sorted arrival times (seconds from the first request) and the token columns."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `python scripts/fetch_datasets.py` first")
    stamps, ctx, gen = [], [], []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            stamps.append(_parse_ts(row["TIMESTAMP"]))
            ctx.append(int(row["ContextTokens"]))
            gen.append(int(row["GeneratedTokens"]))
    t0 = min(stamps)
    t = np.array([(s - t0).total_seconds() for s in stamps])
    order = np.argsort(t, kind="stable")
    return t[order], {"context_tokens": np.array(ctx)[order], "generated_tokens": np.array(gen)[order]}


def rescale_arrivals(times: np.ndarray, rate: float, n: int | None = None, offset_fraction: float = 0.0) -> np.ndarray:
    """Real arrival times rescaled to mean ``rate``, optionally rotated by ``offset_fraction`` of the trace.

    Rotation (wrap-around) gives seed-to-seed variation in where the bursts fall without
    changing the trace's statistics; ``n`` keeps only the first ``n`` arrivals after rotation.
    """
    if rate <= 0:
        raise ValueError("rate must be positive")
    t = np.asarray(times, dtype=float) - float(times[0])
    span = float(t[-1]) + float(np.mean(np.diff(t)))  # period of the wrapped trace
    if offset_fraction:
        t = np.sort((t - offset_fraction * span) % span)
        t = t - t[0]
    if n is not None:
        t = t[:n]
    native_rate = (len(t) - 1) / float(t[-1]) if t[-1] > 0 else float("inf")
    return t * (native_rate / rate)


def arrival_statistics(times: np.ndarray, windows: tuple[float, ...] = (1.0, 10.0, 60.0, 300.0)) -> dict[str, Any]:
    """Burstiness of an arrival stream: interarrival SCV and lag-1 correlation, index of dispersion of counts.

    A Poisson process has SCV 1, zero autocorrelation and IDC 1 at every window.
    """
    t = np.sort(np.asarray(times, dtype=float))
    ia = np.diff(t)
    duration = float(t[-1] - t[0])
    idc = {}
    for w in windows:
        if duration >= 10 * w:
            counts = np.histogram(t, bins=np.arange(t[0], t[-1], w))[0]
            idc[str(w)] = float(counts.var() / counts.mean())
    return {
        "n_arrivals": int(t.size),
        "duration_s": duration,
        "mean_rate": float((t.size - 1) / duration),
        "interarrival_scv": float(ia.var() / ia.mean() ** 2),
        "interarrival_lag1_corr": float(np.corrcoef(ia[:-1], ia[1:])[0, 1]),
        "index_of_dispersion": idc,
    }


def rate_surges(times: np.ndarray, windows: tuple[float, ...] = (60.0, 300.0), factor: float = 1.25) -> dict[str, Any]:
    """Peak-to-mean rate and the fraction of windows whose rate exceeds ``factor`` x the mean.

    At a target utilization ``rho``, windows above ``1/rho`` x the mean rate overload the pool
    (``factor = 1.25`` corresponds to ``rho = 0.8``).
    """
    t = np.sort(np.asarray(times, dtype=float))
    mean = (t.size - 1) / float(t[-1] - t[0])
    out: dict[str, Any] = {}
    for w in windows:
        counts = np.histogram(t, bins=np.arange(t[0], t[-1], w))[0] / w
        if counts.size:
            out[str(w)] = {
                "peak_to_mean": float(counts.max() / mean),
                "fraction_above": float((counts > factor * mean).mean()),
            }
    return {"factor": factor, "windows": out}
