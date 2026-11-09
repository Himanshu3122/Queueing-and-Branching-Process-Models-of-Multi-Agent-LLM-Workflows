"""Mooncake request traces (Qin et al., FAST 2025, Best Paper), Moonshot AI's Kimi service.

Files from ``github.com/kvcache-ai/Mooncake/FAST25-release/traces`` (Apache-2.0). Each line is one
request: ``timestamp`` (ms), ``input_length``, ``output_length`` and ``hash_ids`` (512-token prefix
block hashes; equal ids mean a reusable prefix). Two real workloads are used, each one hour:

* ``toolagent_trace.jsonl`` - 23,608 requests from tool-use and agent applications;
* ``conversation_trace.jsonl`` - 12,031 chat requests.

(``synthetic_trace.jsonl`` has Poisson-generated arrivals and is not real data.)

**Timestamp resolution.** Timestamps are binned to 3-second slots (about 20 requests share each
value in the tool-agent trace). :func:`load_mooncake` spreads the requests of a slot evenly over the
slot, so burstiness below ~3 s is not observable; at 10 s and longer scales the trace is exact.

**Sessions.** The traces are sampled from production traffic, so consecutive calls of one agent
session are mostly absent. Linking requests by prefix hashes connects only ~17% of tool-agent
requests, with a median gap of ~2 minutes, so sessions are not reconstructed (see
:func:`prefix_linkage`).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

SLOT_SECONDS = 3.0


def read_mooncake(path: str | Path) -> list[dict[str, Any]]:
    """Raw request records (one dict per line)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `python scripts/fetch_datasets.py` first")
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_mooncake(path: str | Path, slot_seconds: float = SLOT_SECONDS) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Arrival times in seconds (requests spread evenly within each timestamp slot) and token columns."""
    rows = read_mooncake(path)
    stamps = np.array([r["timestamp"] for r in rows], dtype=float) / 1000.0
    order = np.argsort(stamps, kind="stable")
    stamps = stamps[order]
    times = stamps.copy()
    uniq, start, counts = np.unique(stamps, return_index=True, return_counts=True)
    for s, i, c in zip(uniq, start, counts, strict=True):
        times[i : i + c] = s + slot_seconds * (np.arange(c) + 0.5) / c
    tokens = {
        "input_tokens": np.array([rows[i]["input_length"] for i in order]),
        "output_tokens": np.array([rows[i]["output_length"] for i in order]),
    }
    return times - times[0], tokens


def timestamp_resolution(path: str | Path) -> dict[str, float]:
    """How coarse the recorded timestamps are."""
    t = np.array([r["timestamp"] for r in read_mooncake(path)], dtype=float)
    uniq = np.unique(t)
    gaps = np.diff(uniq)
    return {
        "n_requests": float(t.size),
        "distinct_timestamps": float(uniq.size),
        "median_gap_between_distinct_ms": float(np.median(gaps)) if gaps.size else 0.0,
        "mean_requests_per_timestamp": float(t.size / uniq.size),
    }


def prefix_linkage(path: str | Path, popular_fraction: float = 0.01) -> dict[str, float]:
    """Try to chain requests into sessions by prefix hashes; report how much links.

    A request is linked to the most recent earlier request whose complete 512-token blocks form a
    prefix of its own blocks, requiring at least one non-popular block (popular blocks, present in
    more than ``popular_fraction`` of requests, are shared system prompts and do not identify a
    session). Also reports the fraction of prompt blocks that repeat an earlier block (prefix reuse).
    """
    rows = read_mooncake(path)
    pop = Counter(h for r in rows for h in set(r["hash_ids"]))
    thr = max(20.0, popular_fraction * len(rows))
    full: dict[tuple[int, ...], int] = {}
    parent = np.full(len(rows), -1)
    seen: set[int] = set()
    reused = total = 0
    for j, r in enumerate(rows):
        h = r["hash_ids"]
        reused += sum(1 for x in h if x in seen)
        total += len(h)
        seen.update(h)
        for n in range(len(h) - 1, 0, -1):
            key = tuple(h[:n])
            if key in full and any(pop[x] < thr for x in key):
                parent[j] = full[key]
                break
        nf = r["input_length"] // 512
        if nf > 0:
            full[tuple(h[:nf])] = j
    t = np.array([r["timestamp"] for r in rows], dtype=float) / 1000.0
    linked = parent >= 0
    gaps = t[linked] - t[parent[linked]]
    return {
        "linked_fraction": float(linked.mean()),
        "median_gap_to_parent_s": float(np.median(gaps)) if gaps.size else float("nan"),
        "prefix_block_reuse_fraction": float(reused / total) if total else 0.0,
        "shared_system_blocks": float(sum(1 for v in pop.values() if v >= thr)),
    }
