"""Recursive sub-agent trees (paper Section 5.6, Eqs. (20)-(23)).

After completing its chain successfully, each agent delegates to ``D ~ Poisson(m)`` sub-agents
running the same kind of chain and completes when all of them complete. Under independence of
sub-trees, the latency CDF of successful trees satisfies the fixed point

    F_T = F_X * dH,   H(t) = exp{ m S (F_T(t) - 1) }                     (Eq. 22)

where ``F_X`` is the successful-chain distribution at the load inflated by ``M = 1/(1 - m P_s)``
and ``dH`` includes the atom ``H(0) = exp(-m S)`` for agents that delegate to no one (it lands
in grid bin 0 through :func:`agentq.grid.pmf_from_cdf_values`).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from agentq.config import SystemConfig
from agentq.grid import convolve, pmf_from_cdf_values, summarize
from agentq.model import ModelOptions, chain_prediction
from agentq.structure import structural_quantities


def agents_per_tree(m: float, chain_success: float) -> float:
    """``M = 1 / (1 - m P_s)`` (Eq. (20)); requires ``m P_s < 1``."""
    if m < 0 or m * chain_success >= 1:
        raise ValueError(f"agents per tree is infinite for m*P_s = {m * chain_success:.3f} >= 1")
    return 1.0 / (1.0 - m * chain_success)


def tree_success_probability(chain_success: float, m: float, iterations: int = 500) -> float:
    """Fixed point ``S = P_s exp(m (S - 1))`` reached from ``S = P_s`` (Eq. (21))."""
    s = chain_success
    for _ in range(iterations):
        s = chain_success * math.exp(m * (s - 1.0))
    return s


@dataclass
class TreePrediction:
    """Prediction for successful sub-agent trees."""

    stable: bool
    agents_per_tree: float
    chain_success: float
    rho_llm: float
    rho_tool: float
    tree_success: float | None = None
    mean: float | None = None
    p50: float | None = None
    p95: float | None = None
    p99: float | None = None
    iterations: int | None = None
    converged: bool | None = None

    def to_dict(self) -> dict[str, float | bool | int | None]:
        return asdict(self)


def predict_tree(
    system: SystemConfig,
    task_rate: float,
    delegation_mean: float,
    options: ModelOptions | None = None,
    tol: float = 1e-7,
    max_iterations: int = 300,
) -> TreePrediction:
    """Latency distribution of successful trees with Poisson(m) delegation (Eqs. (20)-(23)).

    Args:
        system: Chain/pool configuration (every agent runs this chain).
        task_rate: Tree (root task) arrival rate.
        delegation_mean: ``m``, mean sub-agents per successful agent.
        options: Model switches for the underlying chain model.
        tol: Sup-norm convergence tolerance on the CDF.
        max_iterations: Iteration cap for the fixed point.
    """
    ps = structural_quantities(system.workflow).success_prob
    mult = agents_per_tree(delegation_mean, ps)
    chain = chain_prediction(system, task_rate, options, multiplier=mult)
    out = TreePrediction(
        stable=chain.stable, agents_per_tree=mult, chain_success=ps, rho_llm=chain.rho_llm, rho_tool=chain.rho_tool
    )
    if not chain.stable or chain.pmf is None:
        return out
    s = tree_success_probability(ps, delegation_mean)
    out.tree_success = s
    f_x = chain.pmf
    cdf = np.cumsum(f_x)
    converged = False
    iterations = 0
    for iterations in range(max_iterations):  # noqa: B007 - final index is reported
        h = pmf_from_cdf_values(np.exp(delegation_mean * s * (cdf - 1.0)))
        new_cdf = np.cumsum(convolve(f_x, h, (options or ModelOptions()).fold_tail))
        if np.max(np.abs(new_cdf - cdf)) < tol:
            cdf = new_cdf
            converged = True
            break
        cdf = new_cdf
    pmf = pmf_from_cdf_values(cdf)
    grid = (options or ModelOptions()).grid
    stats = summarize(pmf, grid)
    out.mean, out.p50, out.p95, out.p99 = stats["mean"], stats["p50"], stats["p95"], stats["p99"]
    out.iterations, out.converged = iterations, converged
    return out
