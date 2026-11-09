"""Azure LLM inference trace 2024 (Stojkovic et al., HPCA 2025, DynamoLLM): one week, two services.

Files ``AzureLLMInferenceTrace_code_1week.csv`` and ``AzureLLMInferenceTrace_conv_1week.csv`` from the
``dataset-llm-2024`` release of ``github.com/Azure/AzurePublicDataset`` (CC-BY 4.0): every request to
a production code-completion and a conversation LLM service over one week of May 2024, with a
microsecond ``TIMESTAMP``, ``ContextTokens`` and ``GeneratedTokens``.

The CSVs are large (0.7 and 1.1 GB), so :func:`load_azure2024` parses them once and caches the
columns as ``.npy`` files next to a chosen cache directory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_azure2024(path: str | Path, cache_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Arrival times (seconds from the first request, sorted), context tokens, generated tokens."""
    path, cache_dir = Path(path), Path(cache_dir)
    stem = cache_dir / path.stem
    files = [stem.with_name(stem.name + f".{k}.npy") for k in ("t", "ctx", "gen")]
    if all(f.is_file() for f in files):
        return tuple(np.load(f) for f in files)  # type: ignore[return-value]
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `python scripts/fetch_datasets.py` first")
    import pyarrow as pa
    from pyarrow import csv as pacsv

    # pyarrow parses the ISO timestamps natively (no per-row Python strings: ~40M rows in total)
    table = pacsv.read_csv(
        path,
        convert_options=pacsv.ConvertOptions(
            column_types={
                "TIMESTAMP": pa.timestamp("us", tz="UTC"),
                "ContextTokens": pa.int32(),
                "GeneratedTokens": pa.int32(),
            }
        ),
    )
    us = table.column("TIMESTAMP").cast(pa.int64()).to_numpy()
    order = np.argsort(us, kind="stable")
    t = (us[order] - us[order][0]) / 1e6
    ctx = table.column("ContextTokens").to_numpy()[order]
    gen = table.column("GeneratedTokens").to_numpy()[order]
    cache_dir.mkdir(parents=True, exist_ok=True)
    for f, arr in zip(files, (t, ctx, gen), strict=True):
        np.save(f, arr)
    return t, ctx, gen


def hourly_surge_index(times: np.ndarray, window: float = 60.0, factor: float = 1.25) -> dict[str, np.ndarray]:
    """Per full hour: share of ``window``-long windows above ``factor`` x the hour's mean, and the peak ratio."""
    t = np.asarray(times, dtype=float)
    per = int(3600 // window)
    counts = np.histogram(t, bins=np.arange(0.0, t[-1] + window, window))[0]
    n_hours = counts.size // per
    blocks = counts[: n_hours * per].reshape(n_hours, per).astype(float)
    mean = blocks.mean(axis=1, keepdims=True)
    ok = mean[:, 0] > 0
    ratio = np.divide(blocks, mean, out=np.zeros_like(blocks), where=mean > 0)
    return {
        "hour": np.arange(n_hours)[ok],
        "requests": blocks.sum(axis=1)[ok],
        "share_above": (ratio > factor).mean(axis=1)[ok],
        "peak_ratio": ratio.max(axis=1)[ok],
    }
