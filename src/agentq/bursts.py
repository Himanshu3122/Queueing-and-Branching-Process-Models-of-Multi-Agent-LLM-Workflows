"""Bursty arrivals: two-state MMPP and the Poisson / quasi-static bracket (paper Section 5.7).

The quasi-static (QS) predictor treats each MMPP state as stationary and mixes the per-state
latency distributions weighted by the fraction of *tasks* arriving in each state (Eq. (25)).
The paper observes, empirically and not as a theorem, that the simulated latency lies between
the Poisson and QS predictions in all 12 tested settings (Eq. (26)).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from agentq.config import SystemConfig
from agentq.grid import summarize
from agentq.model import ModelOptions, chain_prediction, predict


@dataclass(frozen=True)
class MMPP2:
    """Two-state Markov-modulated Poisson process with mean rate ``rate``.

    Attributes:
        rate: Long-run mean arrival rate ``lambda``.
        burst_factor: ``beta = r_1 / r_0`` (high-state rate over low-state rate).
        high_fraction: Long-run fraction of time ``f`` in the high state (paper: 0.25).
        high_exit_rate: Rate ``sigma`` of leaving the high state (mean high sojourn ``1/sigma``).
    """

    rate: float
    burst_factor: float
    high_fraction: float = 0.25
    high_exit_rate: float = 0.02

    def __post_init__(self) -> None:
        if self.rate <= 0 or self.burst_factor < 1:
            raise ValueError("need rate > 0 and burst_factor >= 1")
        if not 0 < self.high_fraction < 1 or self.high_exit_rate <= 0:
            raise ValueError("need 0 < high_fraction < 1 and high_exit_rate > 0")

    @property
    def low_rate(self) -> float:
        """``r_0 = lambda / (1 - f + beta f)`` (Eq. (24))."""
        return self.rate / (1.0 - self.high_fraction + self.burst_factor * self.high_fraction)

    @property
    def high_rate(self) -> float:
        """``r_1 = beta r_0`` (Eq. (24))."""
        return self.burst_factor * self.low_rate

    @property
    def low_exit_rate(self) -> float:
        """Rate of leaving the low state so that the time fraction in high is ``f``."""
        return self.high_exit_rate * self.high_fraction / (1.0 - self.high_fraction)

    @property
    def high_task_weight(self) -> float:
        """Fraction of tasks that arrive in the high state, ``w_1 = f beta / (1 - f + beta f)``."""
        f, b = self.high_fraction, self.burst_factor
        return f * b / (1.0 - f + b * f)


@dataclass
class BracketPrediction:
    """Poisson (optimistic) and quasi-static (pessimistic) predictions for MMPP arrivals."""

    poisson: dict[str, float]
    quasi_static: dict[str, float] | None
    rho_llm_high_state: float
    saturates_in_burst: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def quasi_static_prediction(
    system: SystemConfig, arrivals: MMPP2, options: ModelOptions | None = None
) -> tuple[dict[str, float] | None, float]:
    """QS predictor (Eq. (25)). Returns ``(stats, rho_llm_in_high_state)``; stats are ``None`` if
    the high-state rate saturates a pool, where the predictor is undefined."""
    grid = (options or ModelOptions()).grid
    low = chain_prediction(system, arrivals.low_rate, options)
    high = chain_prediction(system, arrivals.high_rate, options)
    if not (low.stable and high.stable) or low.pmf is None or high.pmf is None:
        return None, high.rho_llm
    w1 = arrivals.high_task_weight
    mix = (1.0 - w1) * low.pmf + w1 * high.pmf
    return summarize(mix, grid), high.rho_llm


def bracket(system: SystemConfig, arrivals: MMPP2, options: ModelOptions | None = None) -> BracketPrediction:
    """Compute the Poisson / quasi-static bracket of Eq. (26)."""
    pois = predict(system, arrivals.rate, options)
    if not pois.stable:
        raise ValueError("system is unstable at the mean arrival rate")
    qs, rho_hi = quasi_static_prediction(system, arrivals, options)
    stats = {"mean": pois.mean, "p50": pois.p50, "p95": pois.p95, "p99": pois.p99}
    return BracketPrediction(
        poisson={k: float(v) for k, v in stats.items() if v is not None},
        quasi_static=qs,
        rho_llm_high_state=rho_hi,
        saturates_in_burst=qs is None,
    )


def interarrival_scv(arrival_times: np.ndarray) -> float:
    """Squared coefficient of variation of interarrival times (used for the ``c_a^2`` correction)."""
    gaps = np.diff(np.asarray(arrival_times, dtype=float))
    if gaps.size < 2:
        raise ValueError("need at least three arrivals")
    return float(gaps.var() / gaps.mean() ** 2)
