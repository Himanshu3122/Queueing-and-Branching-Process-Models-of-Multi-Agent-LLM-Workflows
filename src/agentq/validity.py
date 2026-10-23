"""Validity-region checks: flag configurations where the paper found the model unreliable.

Every threshold below is tied to a reported result in the paper (journal version) and is an
engineering inference (see docs/assumptions.md, A9). Flags are warnings, not hard errors:
predictions are still returned so users can inspect them, but planners should apply a margin or
validate against simulation/traces when any flag is raised.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from agentq.config import SystemConfig
from agentq.structure import pool_loads


@dataclass(frozen=True)
class ValidityFlag:
    """A single warning with its evidence."""

    code: str
    message: str
    evidence: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def check_validity(
    system: SystemConfig,
    task_rate: float,
    delegation_mean: float = 0.0,
    arrival_scv: float | None = None,
) -> list[ValidityFlag]:
    """Return the validity flags raised for a configuration and load."""
    flags: list[ValidityFlag] = []
    loads = pool_loads(system, task_rate)
    rho_max = max(loads.rho_llm, loads.rho_tool)
    eps = 1e-9  # inclusive thresholds, tolerant to round-off when rho is set exactly (e.g. 0.8)
    if rho_max >= 1.0:
        flags.append(ValidityFlag("unstable", "A pool is saturated (rho >= 1); no steady state exists.", "Eq. (10)"))
        return flags
    if rho_max >= 0.9 - eps:
        flags.append(
            ValidityFlag(
                "near_saturation",
                f"Max pool utilization {rho_max:.2f} >= 0.9.",
                "Table 5: p99 underestimated by up to 32.2% at rho_L = 0.9",
            )
        )
    elif rho_max >= 0.8 - eps:
        flags.append(
            ValidityFlag(
                "high_utilization",
                f"Max pool utilization {rho_max:.2f} >= 0.8.",
                "Table 5: errors up to 9.6% (p99) at rho_L = 0.8, growing with load",
            )
        )
    if loads.rho_tool > loads.rho_llm and loads.rho_tool > 0.5:
        flags.append(
            ValidityFlag(
                "tool_bound",
                "The tool pool is the bottleneck.",
                "Table 6: tool-bound P2 had -21.6% mean and -39.6% p99 error",
            )
        )
    if not 0.5 <= system.llm.cv <= 1.5:
        flags.append(
            ValidityFlag(
                "llm_cv_outside_tested",
                f"LLM service CV {system.llm.cv} outside [0.5, 1.5].",
                "Table 8: mean error +31.0% at CV 2.5; error sign changes with CV",
            )
        )
    if system.tool.cv > 1.5:
        flags.append(
            ValidityFlag(
                "tool_cv_untested",
                f"Tool service CV {system.tool.cv} > 1.5 was never tested.",
                "Discussion: measured tool tails are much heavier than CV 1.5",
            )
        )
    if delegation_mean > 0.5:
        flags.append(
            ValidityFlag(
                "delegation_outside_tested",
                f"Delegation m = {delegation_mean} > 0.5.",
                "Table 7 tested m <= 0.5; tail error grows with m",
            )
        )
    if arrival_scv is not None and arrival_scv > 1.2:
        flags.append(
            ValidityFlag(
                "bursty_arrivals",
                f"Interarrival SCV {arrival_scv:.2f} > 1.2.",
                "Table 12: Poisson model off by >2x under MMPP bursts; use the bracket",
            )
        )
    return flags
