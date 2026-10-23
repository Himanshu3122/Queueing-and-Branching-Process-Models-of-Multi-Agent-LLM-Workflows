"""Arrival processes for the reference simulator."""

from __future__ import annotations

import numpy as np

from agentq.bursts import MMPP2


def poisson_arrivals(rng: np.random.Generator, rate: float, n: int) -> np.ndarray:
    """``n`` Poisson arrival times with the given rate."""
    if rate <= 0 or n < 1:
        raise ValueError("need rate > 0 and n >= 1")
    return np.cumsum(rng.exponential(1.0 / rate, n))


def mmpp_arrivals(rng: np.random.Generator, process: MMPP2, horizon: float) -> np.ndarray:
    """Arrival times of a two-state MMPP on ``[0, horizon)``, starting in the low state.

    Sojourns are exponential (high: rate ``sigma``; low: rate ``sigma f / (1 - f)``); within a
    sojourn of length ``d`` the count is Poisson(rate * d) with uniformly placed points. The draw
    order matches the paper's reference implementation so that seeds reproduce its results.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    exit_rate = {1: process.high_exit_rate, 0: process.low_exit_rate}
    rates = {1: process.high_rate, 0: process.low_rate}
    t, state = 0.0, 0
    out: list[float] = []
    while t < horizon:
        d = rng.exponential(1.0 / exit_rate[state])
        end = min(t + d, horizon)
        n = rng.poisson(rates[state] * (end - t))
        out.extend(t + np.sort(rng.random(n)) * (end - t))
        t, state = end, 1 - state
    return np.array(out)
