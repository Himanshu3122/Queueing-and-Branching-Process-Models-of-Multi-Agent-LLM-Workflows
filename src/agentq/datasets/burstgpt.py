"""BurstGPT (Wang et al., KDD 2025): months of Azure OpenAI GPT traffic with failures and latency.

Release ``v2.0`` of ``github.com/HPMLL/BurstGPT`` (CC-BY-4.0). Columns: ``Timestamp`` (whole seconds
from the first day), ``Model`` (``ChatGPT`` = GPT-3.5, ``GPT-4``), ``Request tokens``,
``Response tokens`` (``0`` marks a failed request), ``Total tokens``, ``Log Type`` (``Conversation
log`` or ``API log``). ``BurstGPT_3.csv`` (110 days) adds ``Session ID`` (conversation mode only) and
``Elapsed time``: the measured time from request submission to the complete response, in whole
seconds (end-to-end, so it includes any queueing inside the service).

Used here: ``BurstGPT_1.csv`` (61 days, 1.43M requests) and ``BurstGPT_3.csv`` (110 days, 5.34M).
Timestamps are spread evenly within each second, like the Mooncake slots.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

COLUMNS = {
    "Timestamp": "t",
    "Session ID": "session",
    "Elapsed time": "elapsed",
    "Model": "model",
    "Request tokens": "req_tokens",
    "Response tokens": "resp_tokens",
    "Log Type": "log_type",
}


def load_burstgpt(path: str | Path):  # -> pandas.DataFrame
    """Load one BurstGPT CSV with short column names, a ``failed`` flag and sub-second spread times."""
    import pandas as pd

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `python scripts/fetch_datasets.py` first")
    df = pd.read_csv(path).rename(columns=COLUMNS)
    df = df[[c for c in COLUMNS.values() if c in df.columns]]
    df = df.sort_values("t", kind="stable").reset_index(drop=True)
    df["failed"] = df["resp_tokens"] == 0
    sec = df["t"].to_numpy(dtype=float)
    rank = df.groupby("t").cumcount().to_numpy()
    size = df.groupby("t")["t"].transform("size").to_numpy()
    spread = sec + (rank + 0.5) / size
    df["time"] = spread - spread[0]
    return df


def windowed_counts(times: np.ndarray, window: float) -> np.ndarray:
    t = np.asarray(times, dtype=float)
    return np.histogram(t, bins=np.arange(t[0], t[-1] + window, window))[0]


def local_surges(times: np.ndarray, window: float = 300.0, base: float = 3600.0, factor: float = 1.25) -> dict:
    """Share of ``window``-long windows whose rate exceeds ``factor`` x the mean of their ``base`` period.

    Comparable with a one-hour trace replayed at its own mean rate (Azure 2023, Mooncake); empty
    base periods are skipped.
    """
    counts = windowed_counts(times, window)
    per = int(base // window)
    n = (counts.size // per) * per
    blocks = counts[:n].reshape(-1, per).astype(float)
    means = blocks.mean(axis=1, keepdims=True)
    ok = means[:, 0] > 0
    ratio = blocks[ok] / means[ok]
    return {
        "window_s": window,
        "base_s": base,
        "fraction_above": float((ratio > factor).mean()),
        "hours_with_any_window_above": float((ratio > factor).any(axis=1).mean()),
        "p99_window_to_hour_ratio": float(np.percentile(ratio, 99)),
    }
