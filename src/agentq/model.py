"""Semi-analytical latency prediction for one agent chain (paper Algorithm 1, Eqs. (6)-(19)).

Pipeline::

    structure (Eqs. 6-8) -> loads (Eq. 9) -> pool response pmfs (Eqs. 11-14)
    -> fork-join (Eq. 15) -> attempt (Eq. 16) -> retry mixture (Eq. 17)
    -> success-tilted step mixture (Eq. 18) -> quantiles (Eq. 19)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from agentq.config import SystemConfig
from agentq.grid import FloatArray, Grid, convolve, pmf_from_cdf_values, summarize
from agentq.queueing import ServiceDist, response_pmf, service_pmf, wait_parameters
from agentq.structure import pool_loads, structural_quantities


@dataclass(frozen=True)
class ModelOptions:
    """Switches for the model and its ablations (paper Table 11).

    Attributes:
        use_queueing: If False, pool response = service time only ("No queueing" ablation).
        use_fork_join: If False, a step waits for a single tool response instead of the
            maximum over ``K`` ("No fork-join" ablation).
        ca2: Squared coefficient of variation of arrivals at each pool (``c_a^2``, Eq. (12)).
        service_dist: ``lognormal`` (paper) or ``exponential`` (true M/M/k service).
        tool_ca2: ``c_a^2`` at the tool pool only, overriding ``ca2`` there. ``"batch"`` uses
            ``E[K^2]/E[K]``, the index of dispersion of a compound-Poisson stream whose batches
            are the forks (siblings arrive together), instead of Poisson arrivals.
        max_steps: Truncation ``n_max`` of the step-count mixture (paper: 80).
        step_tail_tol: If set, ``n_max`` is raised until the dropped success-tilted step-count
            mass ``((1-q) s)^{n_max}`` is below this tolerance (the paper's truncation at 80
            loses about 3e-3 of the mass at ``n_bar = 16``, biasing p99 down).
        fold_tail: Fold mass beyond the grid horizon into the last bin instead of dropping it
            and renormalizing (see :func:`agentq.grid.convolve`).
        grid: Discretization grid (paper: dt = 0.02 s, 30,000 points).
    """

    use_queueing: bool = True
    use_fork_join: bool = True
    ca2: float = 1.0
    tool_ca2: float | str | None = None
    service_dist: ServiceDist = "lognormal"
    max_steps: int = 80
    step_tail_tol: float | None = None
    fold_tail: bool = False
    grid: Grid = field(default_factory=Grid)

    def steps_needed(self, ratio: float) -> int:
        """``n_max`` for a geometric step-count tail ``ratio^n`` (``ratio = (1-q) s``)."""
        if self.step_tail_tol is None or ratio <= 0.0:
            return self.max_steps
        return max(self.max_steps, int(np.ceil(np.log(self.step_tail_tol) / np.log(ratio))))


REVISED_OPTIONS = {"step_tail_tol": 1e-10, "fold_tail": True}
"""Numerical settings of the revised manuscript (no step truncation bias, no horizon loss)."""


@dataclass
class Prediction:
    """Model output for one configuration and arrival rate.

    Latency fields are for *successful* tasks and are ``None`` when the system is unstable.
    """

    stable: bool
    rho_llm: float
    rho_tool: float
    success_prob: float
    llm_calls_per_task: float
    calls_per_success: float
    started_steps: float
    mean: float | None = None
    p50: float | None = None
    p95: float | None = None
    p99: float | None = None
    horizon_tail_mass: float | None = None
    pmf: FloatArray | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, float | bool | None]:
        """JSON-friendly dict without the pmf array."""
        d = asdict(self)
        d.pop("pmf")
        return d


def _join_pmf(tool_response: FloatArray, fanout: float, kind: str) -> FloatArray:
    """Pmf of the join time ``F_J = G(F_R^T)`` (Eq. (15)).

    Fixed ``K``: ``F_J = F^K``; Poisson ``K``: ``F_J = exp(mu_K (F - 1))`` (an empty fork,
    ``K = 0``, contributes an atom at zero).
    """
    cdf = np.cumsum(tool_response)
    joined = cdf**fanout if kind == "fixed" else np.exp(fanout * (cdf - 1.0))
    return pmf_from_cdf_values(joined)


def _mixture_of_powers(base: FloatArray, weights: FloatArray, fold_tail: bool = False) -> FloatArray:
    """``sum_i weights[i] * base^{*(i+1)}`` using repeated convolution."""
    out = np.zeros_like(base)
    cur = base.copy()
    for i, w in enumerate(weights):
        out += w * cur
        if i < len(weights) - 1:
            cur = convolve(cur, base, fold_tail)
    return out


def tool_arrival_scv(fanout: float, kind: str) -> float:
    """``E[K^2]/E[K]`` for fixed (``K``) or Poisson (``1 + mu_K``) fan-out."""
    return fanout if kind == "fixed" else 1.0 + fanout


def chain_prediction(
    system: SystemConfig,
    task_rate: float,
    options: ModelOptions | None = None,
    multiplier: float = 1.0,
) -> Prediction:
    """Predict the latency distribution of successful chains.

    Args:
        system: Workflow and pool configuration.
        task_rate: External task arrival rate ``lambda`` (tasks/s).
        options: Model switches; defaults reproduce the paper's full model.
        multiplier: Scales the chain arrival rate at the pools (agents per tree ``M``, Eq. (23)).

    Returns:
        A :class:`Prediction`; ``stable`` is False and latencies are ``None`` if any pool has
        ``rho >= 1``.
    """
    opts = options or ModelOptions()
    if task_rate <= 0:
        raise ValueError(f"task_rate must be positive, got {task_rate}")
    wf = system.workflow
    sq = structural_quantities(wf)
    loads = pool_loads(system, task_rate, multiplier)
    pred = Prediction(
        stable=loads.stable,
        rho_llm=loads.rho_llm,
        rho_tool=loads.rho_tool,
        success_prob=sq.success_prob,
        llm_calls_per_task=sq.llm_calls_per_task,
        calls_per_success=sq.calls_per_success,
        started_steps=sq.started_steps,
    )
    if not loads.stable:
        return pred

    grid = opts.grid
    exp_service = opts.service_dist == "exponential"
    cv_l = 1.0 if exp_service else system.llm.cv
    cv_t = 1.0 if exp_service else system.tool.cv
    if opts.tool_ca2 == "batch":
        ca2_t = tool_arrival_scv(wf.fanout, wf.fanout_kind)
    elif opts.tool_ca2 is not None:
        ca2_t = float(opts.tool_ca2)
    else:
        ca2_t = opts.ca2
    fold = opts.fold_tail
    if opts.use_queueing:
        r_l = response_pmf(
            loads.llm_arrival_rate, system.llm.mean_service, cv_l, system.llm.servers, grid, opts.ca2, opts.service_dist
        )
        r_t = response_pmf(
            loads.tool_arrival_rate,
            system.tool.mean_service,
            cv_t,
            system.tool.servers,
            grid,
            ca2_t,
            opts.service_dist,
        )
        if r_l is None or r_t is None:  # pragma: no cover - excluded by the stability check above
            pred.stable = False
            return pred
    else:
        r_l = service_pmf(system.llm.mean_service, cv_l, grid, opts.service_dist)
        r_t = service_pmf(system.tool.mean_service, cv_t, grid, opts.service_dist)

    if wf.fanout == 0:
        attempt = r_l
    elif opts.use_fork_join:
        attempt = convolve(r_l, _join_pmf(r_t, wf.fanout, wf.fanout_kind), fold)
    else:
        attempt = convolve(r_l, r_t, fold)

    p, r = wf.failure_prob, wf.max_retries
    w_att = np.array([p**i * (1.0 - p) for i in range(r + 1)])
    w_att /= w_att.sum()
    step = _mixture_of_powers(attempt, w_att, fold)

    q, s = wf.q, sq.step_success
    n = np.arange(1, opts.steps_needed((1.0 - q) * s) + 1)
    w_steps = q * (1.0 - q) ** (n - 1) * s**n
    w_steps /= w_steps.sum()
    task = _mixture_of_powers(step, w_steps, fold)

    stats = summarize(task, grid)
    pred.mean, pred.p50, pred.p95, pred.p99 = stats["mean"], stats["p50"], stats["p95"], stats["p99"]
    pred.horizon_tail_mass = float(task[-1])
    pred.pmf = task
    return pred


def predict(system: SystemConfig, task_rate: float, options: ModelOptions | None = None) -> Prediction:
    """Full-model prediction for a single chain (paper Algorithm 1)."""
    return chain_prediction(system, task_rate, options)


def mean_value_baseline(system: SystemConfig, task_rate: float) -> float | None:
    """Mean-value baseline of paper Table 11 ("Mean-value (no distribution)").

    Sums mean waits and mean service times per attempt, multiplied by expected attempts:
    ``nu * a * (E[S_L] + E[W_L] + E[S_T] + E[W_T])``. As in the paper's implementation, one
    tool response per attempt is counted (no fan-out maximum) and latency is not conditioned
    on success. Returns ``None`` if unstable.
    """
    sq = structural_quantities(system.workflow)
    loads = pool_loads(system, task_rate)
    w_l = wait_parameters(loads.llm_arrival_rate, system.llm.mean_service, system.llm.cv, system.llm.servers)
    w_t = wait_parameters(loads.tool_arrival_rate, system.tool.mean_service, system.tool.cv, system.tool.servers)
    if w_l is None or w_t is None:
        return None
    per_attempt = system.llm.mean_service + w_l.mean_wait + system.tool.mean_service + w_t.mean_wait
    return sq.started_steps * sq.attempts_per_step * per_attempt
