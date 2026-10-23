"""Single-pool (G/G/k) response-time approximation (paper Section 5.4, Eqs. (11)-(14)).

Each pool (LLM endpoints, tool workers) is a FIFO multi-server queue. The waiting time is
approximated by

* the Erlang-C probability of waiting ``C(k, alpha)`` (Eq. (11)),
* a two-moment mean wait that scales the M/M/k mean wait by ``(c_a^2 + c_s^2)/2`` (Eq. (12)),
* a wait distribution that is an atom at zero with mass ``1 - C`` plus an exponential with
  the matched conditional mean (Eq. (13)),

and the response time is wait plus service, obtained by convolution (Eq. (14)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import lognorm

from agentq.grid import FloatArray, Grid, convolve

ServiceDist = Literal["lognormal", "exponential"]


def lognormal_params(mean: float, cv: float) -> tuple[float, float]:
    """Return ``(mu, sigma)`` of a lognormal with the given mean and coefficient of variation."""
    if mean <= 0 or cv <= 0:
        raise ValueError(f"lognormal needs mean > 0 and cv > 0, got mean={mean}, cv={cv}")
    s2 = math.log(1.0 + cv * cv)
    return math.log(mean) - s2 / 2.0, math.sqrt(s2)


def erlang_c(servers: int, offered_load: float) -> float:
    """Erlang-C probability that an arrival waits in an M/M/k queue (Eq. (11)).

    Computed with the numerically stable Erlang-B recursion
    ``B_j = a B_{j-1} / (j + a B_{j-1})`` and ``C = B_k / (1 - rho (1 - B_k))``,
    which is algebraically identical to the factorial form in the paper and does not
    overflow for large ``k``.

    Args:
        servers: Number of servers ``k`` (>= 1).
        offered_load: ``alpha = lambda * E[S]``.

    Returns:
        Probability of waiting; 1.0 when ``alpha >= k`` (unstable).
    """
    if servers < 1:
        raise ValueError(f"servers must be >= 1, got {servers}")
    if offered_load < 0:
        raise ValueError(f"offered load must be >= 0, got {offered_load}")
    rho = offered_load / servers
    if rho >= 1.0:
        return 1.0
    if offered_load == 0:
        return 0.0
    b = 1.0
    for j in range(1, servers + 1):
        b = offered_load * b / (j + offered_load * b)
    return b / (1.0 - rho * (1.0 - b))


@dataclass(frozen=True)
class WaitParameters:
    """Parameters of the approximate wait distribution of one pool.

    Attributes:
        prob_wait: Probability ``C`` that a request waits at all.
        conditional_mean: Mean wait given that the request waits (``E[W] / C``).
        mean_wait: Unconditional mean wait ``E[W]`` (Eq. (12)).
        utilization: ``rho = lambda E[S] / k``.
    """

    prob_wait: float
    conditional_mean: float
    mean_wait: float
    utilization: float


def wait_parameters(
    arrival_rate: float,
    mean_service: float,
    cv_service: float,
    servers: int,
    ca2: float = 1.0,
) -> WaitParameters | None:
    """Two-moment wait approximation for a G/G/k pool (Eqs. (11)-(12)).

    Returns ``None`` when the pool is unstable (``rho >= 1``).
    """
    if arrival_rate < 0 or mean_service <= 0 or cv_service < 0 or ca2 < 0:
        raise ValueError("invalid wait parameters")
    offered = arrival_rate * mean_service
    rho = offered / servers
    if rho >= 1.0:
        return None
    c = erlang_c(servers, offered)
    if c == 0.0:
        return WaitParameters(0.0, 0.0, 0.0, rho)
    mean_w = c / (servers / mean_service - arrival_rate) * (ca2 + cv_service**2) / 2.0
    return WaitParameters(c, mean_w / c, mean_w, rho)


def service_pmf(mean: float, cv: float, grid: Grid, dist: ServiceDist = "lognormal") -> FloatArray:
    """Grid pmf of a service time.

    ``lognormal`` is the paper's assumption (Section 3.1). ``exponential`` ignores ``cv`` and is
    provided for a true M/M/k-service ablation (see docs/assumptions.md, A6).
    """
    edges = grid.bin_edges()
    if dist == "lognormal":
        mu, sigma = lognormal_params(mean, cv)
        cdf = lognorm.cdf(edges, s=sigma, scale=math.exp(mu))
    elif dist == "exponential":
        if mean <= 0:
            raise ValueError("exponential service needs mean > 0")
        cdf = 1.0 - np.exp(-edges / mean)
    else:  # pragma: no cover - guarded by typing and config validation
        raise ValueError(f"unknown service distribution: {dist}")
    return grid.pmf_from_cdf(cdf)


def wait_pmf(params: WaitParameters, grid: Grid) -> FloatArray:
    """Atom at zero with mass ``1 - C`` plus ``C`` times an exponential (Eq. (13))."""
    pmf = np.zeros(grid.n)
    pmf[0] = 1.0 - params.prob_wait
    if params.prob_wait > 0:
        edges = grid.bin_edges()
        cdf = 1.0 - np.exp(-edges / params.conditional_mean)
        pmf += params.prob_wait * grid.pmf_from_cdf(cdf)
    return pmf


def response_pmf(
    arrival_rate: float,
    mean_service: float,
    cv_service: float,
    servers: int,
    grid: Grid,
    ca2: float = 1.0,
    dist: ServiceDist = "lognormal",
) -> FloatArray | None:
    """Response-time pmf ``F_R = F_W * F_S`` of one pool (Eq. (14)); ``None`` if unstable."""
    params = wait_parameters(arrival_rate, mean_service, cv_service, servers, ca2)
    if params is None:
        return None
    return convolve(wait_pmf(params, grid), service_pmf(mean_service, cv_service, grid, dist))
