"""Berkeley Function Calling Leaderboard (Patil et al., ICML 2025) as a real tool fan-out law.

Files from ``gorilla-llm/Berkeley-Function-Calling-Leaderboard`` on the Hugging Face Hub
(Apache-2.0). The fan-out ``K`` of a query is the number of function calls in its ground-truth
answer (``possible_answer/*.json``).

* **BFCL-Live** (``live_simple``, ``live_multiple``, ``live_parallel``, ``live_parallel_multiple``)
  is built from real user queries contributed by enterprises and open-source users. Its category
  sizes are the only ones that approximate how often real queries fork. (They are still curated:
  BFCL deduplicated and filtered the contributions.)
* The designed categories (``parallel``, ``parallel_multiple``) were written by the authors; their
  ``K`` values show what fan-outs occur, not how often.

``live_irrelevance`` queries (no function should be called) are excluded from the fan-out law: they
would add ``K = 0`` attempts whose share is a benchmark design choice.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

LIVE = ("live_simple", "live_multiple", "live_parallel", "live_parallel_multiple")
LIVE_PARALLEL = ("live_parallel", "live_parallel_multiple")
DESIGNED_PARALLEL = ("parallel", "parallel_multiple")


def fanout_counts(data_dir: str | Path, categories: tuple[str, ...]) -> np.ndarray:
    """Number of ground-truth calls per query over the given categories."""
    data_dir = Path(data_dir)
    out: list[int] = []
    for cat in categories:
        path = data_dir / "possible_answer" / f"BFCL_v3_{cat}.json"
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found; run `python scripts/fetch_datasets.py` first")
        with path.open(encoding="utf-8") as fh:
            out.extend(len(json.loads(line)["ground_truth"]) for line in fh if line.strip())
    return np.array(out, dtype=int)


def empirical_pmf(counts: np.ndarray) -> dict[int, float]:
    """``{k: P(K = k)}`` from observed counts."""
    c = Counter(int(k) for k in counts)
    n = sum(c.values())
    return {k: v / n for k, v in sorted(c.items())}
