"""Closed-form structural quantities (paper Sections 5.2-5.3, Eqs. (6)-(10)).

These depend only on the workflow structure and means; no queueing approximation is involved.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from agentq.config import SystemConfig, WorkflowConfig


@dataclass(frozen=True)
class StructuralQuantities:
    """Retry and step accounting for one chain.

    Attributes:
        step_success: ``s = 1 - p^(R+1)`` (Eq. (6)).
        attempts_per_step: ``a = (1 - p^(R+1)) / (1 - p)`` (Eq. (6)).
        started_steps: ``nu = 1 / (1 - (1 - q) s)`` (Eq. (7)).
        success_prob: ``P_s = q s nu`` (Eq. (7)).
        llm_calls_per_task: ``c_L = nu a``.
        calls_per_success: ``C_s = c_L / P_s = n_bar / (1 - p)`` (Eq. (8)).
        mean_fanout: ``E[K]``.
        tool_calls_per_task: ``c_L E[K]``.
    """

    step_success: float
    attempts_per_step: float
    started_steps: float
    success_prob: float
    llm_calls_per_task: float
    calls_per_success: float
    mean_fanout: float
    tool_calls_per_task: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def structural_quantities(workflow: WorkflowConfig) -> StructuralQuantities:
    """Compute Eqs. (6)-(8) for a workflow."""
    p, r, q = workflow.failure_prob, workflow.max_retries, workflow.q
    s = 1.0 - p ** (r + 1)
    a = (1.0 - p ** (r + 1)) / (1.0 - p) if p > 0 else 1.0
    nu = 1.0 / (1.0 - (1.0 - q) * s)
    ps = q * s * nu
    c_l = nu * a
    k = float(workflow.fanout)
    return StructuralQuantities(
        step_success=s,
        attempts_per_step=a,
        started_steps=nu,
        success_prob=ps,
        llm_calls_per_task=c_l,
        calls_per_success=c_l / ps,
        mean_fanout=k,
        tool_calls_per_task=c_l * k,
    )


@dataclass(frozen=True)
class PoolLoads:
    """Per-pool arrival rates and utilizations (Eq. (9))."""

    llm_arrival_rate: float
    tool_arrival_rate: float
    rho_llm: float
    rho_tool: float

    @property
    def stable(self) -> bool:
        return self.rho_llm < 1.0 and self.rho_tool < 1.0

    @property
    def bottleneck(self) -> str:
        return "llm" if self.rho_llm >= self.rho_tool else "tool"


def pool_loads(system: SystemConfig, task_rate: float, multiplier: float = 1.0) -> PoolLoads:
    """Offered load on both pools for task arrival rate ``lambda`` (Eq. (9)).

    ``multiplier`` scales the chain arrival rate, e.g. by the agents-per-tree factor ``M``
    (Eq. (23)).
    """
    if task_rate < 0 or multiplier <= 0:
        raise ValueError("task_rate must be >= 0 and multiplier > 0")
    sq = structural_quantities(system.workflow)
    lam_l = task_rate * multiplier * sq.llm_calls_per_task
    lam_t = lam_l * sq.mean_fanout
    return PoolLoads(
        llm_arrival_rate=lam_l,
        tool_arrival_rate=lam_t,
        rho_llm=lam_l * system.llm.mean_service / system.llm.servers,
        rho_tool=lam_t * system.tool.mean_service / system.tool.servers,
    )


@dataclass(frozen=True)
class StabilityLimit:
    """Maximum sustainable task rate and the pool that attains it (Eq. (10))."""

    llm_limit: float
    tool_limit: float

    @property
    def max_rate(self) -> float:
        return min(self.llm_limit, self.tool_limit)

    @property
    def bottleneck(self) -> str:
        return "llm" if self.llm_limit <= self.tool_limit else "tool"


def stability_limit(system: SystemConfig, multiplier: float = 1.0) -> StabilityLimit:
    """``lambda* = min{k_L / (c_L E[S_L]), k_T / (c_L E[K] E[S_T])}`` (Eq. (10))."""
    sq = structural_quantities(system.workflow)
    c_l = sq.llm_calls_per_task * multiplier
    llm = system.llm.servers / (c_l * system.llm.mean_service)
    tool_calls = c_l * sq.mean_fanout
    tool = float("inf") if tool_calls == 0 else system.tool.servers / (tool_calls * system.tool.mean_service)
    return StabilityLimit(llm_limit=llm, tool_limit=tool)


def rate_for_llm_utilization(system: SystemConfig, rho: float, multiplier: float = 1.0) -> float:
    """Task arrival rate that produces LLM utilization ``rho`` (used throughout Section 7)."""
    if not 0 < rho < 1:
        raise ValueError(f"target utilization must be in (0, 1), got {rho}")
    sq = structural_quantities(system.workflow)
    return rho * system.llm.servers / (system.llm.mean_service * sq.llm_calls_per_task * multiplier)
