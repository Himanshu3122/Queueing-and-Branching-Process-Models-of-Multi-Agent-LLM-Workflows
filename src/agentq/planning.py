"""SLO-driven capacity planning with a validation margin (paper Section 5.8, Eq. (27)).

``lambda_SLO(delta) = max{ lambda < lambda* : t_0.99(lambda) <= delta * tau }``

Predicted p99 is increasing in ``lambda``, so the maximum is found by bisection. ``delta < 1``
derates the target to compensate for the model's known p99 underestimation near saturation;
the paper evaluates ``delta = 0.85`` (Table 10). With ``delta = 1`` the simulated p99 exceeded
the target in all four tested pool sizes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from agentq.config import SystemConfig
from agentq.model import ModelOptions, predict
from agentq.structure import stability_limit


@dataclass(frozen=True)
class CapacityResult:
    """Outcome of an SLO capacity search."""

    max_rate: float
    target_p99: float
    derating: float
    predicted_p99: float | None
    stability_limit: float
    bottleneck: str

    def to_dict(self) -> dict[str, float | str | None]:
        return asdict(self)


def _p99(system: SystemConfig, rate: float, options: ModelOptions | None) -> float:
    pred = predict(system, rate, options)
    return pred.p99 if pred.stable and pred.p99 is not None else float("inf")


def max_rate_for_p99(
    system: SystemConfig,
    target_p99: float,
    derating: float = 0.85,
    options: ModelOptions | None = None,
    iterations: int = 40,
    lower: float = 1e-3,
) -> CapacityResult:
    """Largest task rate whose predicted p99 is at most ``derating * target_p99`` (Eq. (27)).

    Bisection on ``[lower, 0.999 * lambda*]`` with ``iterations`` halvings, as in the paper.
    Returns ``max_rate = lower`` if even the lower bound violates the derated target.
    """
    if target_p99 <= 0 or not 0 < derating <= 1:
        raise ValueError("need target_p99 > 0 and 0 < derating <= 1")
    limit = stability_limit(system)
    lo, hi = lower, 0.999 * limit.max_rate
    if hi <= lo:
        raise ValueError("stability limit is below the search lower bound")
    goal = derating * target_p99
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if _p99(system, mid, options) <= goal:
            lo = mid
        else:
            hi = mid
    p99 = _p99(system, lo, options)
    return CapacityResult(
        max_rate=lo,
        target_p99=target_p99,
        derating=derating,
        predicted_p99=None if p99 == float("inf") else p99,
        stability_limit=limit.max_rate,
        bottleneck=limit.bottleneck,
    )


def min_llm_servers_for_rate(
    system: SystemConfig,
    task_rate: float,
    target_p99: float,
    derating: float = 0.85,
    max_servers: int = 1024,
    tool_ratio: float | None = None,
    options: ModelOptions | None = None,
) -> int | None:
    """Smallest ``k_L`` whose derated predicted p99 meets the target at ``task_rate``.

    If ``tool_ratio`` is given, the tool pool is scaled as ``k_T = tool_ratio * k_L`` (the paper's
    capacity experiment uses ``k_T = 2 k_L``). Returns ``None`` if no size up to ``max_servers``
    works. This search is an engineering extension; the paper only evaluates the rate search.
    """
    goal = derating * target_p99
    for k in range(1, max_servers + 1):
        updates: dict[str, int] = {"llm.servers": k}
        if tool_ratio is not None:
            updates["tool.servers"] = max(1, round(tool_ratio * k))
        candidate = system.with_updates(**updates)
        if _p99(candidate, task_rate, options) <= goal:
            return k
    return None
