"""Discretized time grid and distribution utilities.

All latency distributions in the model are probability mass functions (pmfs) on a
uniform grid ``t_i = i * dt`` for ``i = 0..n-1`` (paper, Section 5.5: ``Delta t = 0.02 s``,
``G = 30,000`` points, i.e. a 600 s horizon). Mass beyond the horizon is folded into the
last bin, so very long tails are truncated, not lost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

PAPER_DT = 0.02
PAPER_POINTS = 30_000


@dataclass(frozen=True)
class Grid:
    """Uniform time grid used for all pmfs.

    Attributes:
        dt: Bin width in seconds.
        n: Number of grid points. The horizon is ``dt * n`` seconds.
    """

    dt: float = PAPER_DT
    n: int = PAPER_POINTS

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError(f"grid dt must be positive, got {self.dt}")
        if self.n < 16:
            raise ValueError(f"grid must have at least 16 points, got {self.n}")

    @property
    def times(self) -> FloatArray:
        """Grid point locations ``t_i = i * dt``."""
        return np.arange(self.n, dtype=float) * self.dt

    @property
    def horizon(self) -> float:
        """Largest representable time in seconds."""
        return self.dt * self.n

    def bin_edges(self) -> FloatArray:
        """Edges ``(i - 1/2) dt`` with the first edge clamped at 0 (point 0 is the atom at 0)."""
        edges = (np.arange(self.n + 1, dtype=float) - 0.5) * self.dt
        edges[0] = 0.0
        return edges

    def pmf_from_cdf(self, cdf_at_edges: FloatArray) -> FloatArray:
        """Convert CDF values at :meth:`bin_edges` into a pmf, folding the tail into the last bin."""
        pmf = np.diff(cdf_at_edges)
        pmf[-1] += 1.0 - cdf_at_edges[-1]
        return pmf


def convolve(a: FloatArray, b: FloatArray, fold_tail: bool = False) -> FloatArray:
    """Distribution of the sum of two independent grid variables (linear FFT convolution).

    The full linear convolution has ``2n - 1`` points. By default (the reference implementation
    used for the paper) the part beyond the horizon is dropped and the rest renormalized, which
    biases tails down when mass reaches the horizon. With ``fold_tail`` that part is added to
    the last bin instead, so ``pmf[-1]`` measures the mass at or beyond the horizon.
    """
    if a.shape != b.shape:
        raise ValueError(f"pmf shapes differ: {a.shape} vs {b.shape}")
    n = a.shape[0]
    size = 2 * n
    full = np.fft.irfft(np.fft.rfft(a, size) * np.fft.rfft(b, size), size)
    out = full[:n].copy()
    if fold_tail:
        out[-1] += max(float(full[n:].sum()), 0.0)
    out = np.maximum(out, 0.0)
    total = out.sum()
    if total <= 0:
        raise ValueError("convolution produced an empty distribution")
    return out / total


def pmf_from_cdf_values(cdf_values: FloatArray) -> FloatArray:
    """Pmf from CDF values sampled at grid points (``F(t_i)``), clipping negatives and renormalizing."""
    pmf = np.diff(np.concatenate([[0.0], cdf_values]))
    pmf = np.maximum(pmf, 0.0)
    return pmf / pmf.sum()


def mean(pmf: FloatArray, grid: Grid) -> float:
    """Mean of a grid pmf."""
    return float((pmf * grid.times).sum())


def quantile(pmf: FloatArray, grid: Grid, level: float) -> float:
    """Smallest grid time ``t`` with ``F(t) >= level`` (paper Eq. (19))."""
    if not 0.0 < level < 1.0:
        raise ValueError(f"quantile level must be in (0, 1), got {level}")
    cdf = np.cumsum(pmf)
    idx = int(np.searchsorted(cdf, level))
    return float(grid.times[min(idx, grid.n - 1)])


def summarize(pmf: FloatArray, grid: Grid) -> dict[str, float]:
    """Mean, median, p95 and p99 of a grid pmf."""
    return {
        "mean": mean(pmf, grid),
        "p50": quantile(pmf, grid, 0.50),
        "p95": quantile(pmf, grid, 0.95),
        "p99": quantile(pmf, grid, 0.99),
    }


def tail_mass_at_horizon(pmf: FloatArray) -> float:
    """Probability mass in the last bin; a large value means the horizon truncates the tail."""
    return float(pmf[-1])
