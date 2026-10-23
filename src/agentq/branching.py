"""Galton-Watson branching results for call and agent spawning (paper Section 5.1, Eqs. (1)-(5))."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq


def expected_total_progeny(m: float) -> float:
    """``E[Z] = 1 / (1 - m)`` for a subcritical tree rooted at one call (Eq. (1))."""
    if not 0 <= m < 1:
        raise ValueError(f"E[Z] is finite only for 0 <= m < 1, got {m}")
    return 1.0 / (1.0 - m)


def variance_total_progeny(m: float, offspring_variance: float) -> float:
    """``Var[Z] = sigma^2 / (1 - m)^3`` (Eq. (2))."""
    if not 0 <= m < 1:
        raise ValueError(f"Var[Z] is finite only for 0 <= m < 1, got {m}")
    if offspring_variance < 0:
        raise ValueError("offspring variance must be >= 0")
    return offspring_variance / (1.0 - m) ** 3


def poisson_extinction_probability(m: float) -> float:
    """Smallest root in ``[0, 1]`` of ``xi = exp(m (xi - 1))`` (Eq. (3) with Poisson offspring)."""
    if m <= 1.0:
        return 1.0
    return float(brentq(lambda x: math.exp(m * (x - 1.0)) - x, 0.0, 0.999999))


def borel_pmf(n: int, m: float) -> float:
    """``P(Z = n) = e^{-mn} (mn)^{n-1} / n!`` for Poisson offspring (Eq. (4))."""
    if n < 1:
        return 0.0
    if m <= 0:
        return 1.0 if n == 1 else 0.0
    return math.exp(-m * n + (n - 1) * math.log(m * n) - math.lgamma(n + 1))


def borel_tail(n: int, m: float) -> float:
    """``P(Z > n)`` under the Borel law."""
    return 1.0 - sum(borel_pmf(k, m) for k in range(1, n + 1))


def criticality_relative_se(m: float, n_obs: int, offspring_sd: float | None = None) -> float:
    """Delta-method relative standard error of the estimated ``E[Z]`` (Eq. (5)).

    ``se(Z_hat)/E[Z] ~= sigma / (sqrt(n) (1 - m))``; ``sigma`` defaults to ``sqrt(m)`` (Poisson).
    """
    if not 0 <= m < 1 or n_obs < 1:
        raise ValueError("need 0 <= m < 1 and n_obs >= 1")
    sigma = math.sqrt(m) if offspring_sd is None else offspring_sd
    return sigma / (math.sqrt(n_obs) * (1.0 - m))


def simulate_total_progeny(
    rng: np.random.Generator,
    m: float,
    n_trees: int,
    cap: int = 5000,
    dispersion: float | None = None,
) -> NDArray[np.float64]:
    """Monte Carlo total progeny of ``n_trees`` Galton-Watson trees.

    Offspring are Poisson(m), or negative binomial with mean ``m`` and dispersion ``r``
    (variance ``m + m^2/r``) when ``dispersion`` is given. Trees are truncated at ``cap``
    nodes, which is how the paper separates bounded from exploding growth.
    """
    if n_trees < 1 or cap < 1:
        raise ValueError("n_trees and cap must be >= 1")
    out = np.empty(n_trees)
    for i in range(n_trees):
        total, alive = 1, 1
        while alive > 0 and total < cap:
            if dispersion is None:
                kids = int(rng.poisson(m, alive).sum())
            else:
                kids = int(rng.negative_binomial(dispersion, dispersion / (dispersion + m), alive).sum())
            total += kids
            alive = kids
        out[i] = total
    return out
