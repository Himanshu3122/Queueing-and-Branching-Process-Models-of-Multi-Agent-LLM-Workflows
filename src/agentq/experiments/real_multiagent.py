"""Multi-agent structure and failures from Who&When (ICML 2025).

Tests three model assumptions on real multi-agent runs (no timings, so no latency):

1. **Independent attempt failures** - code-execution failures of CaptainAgent teams:
   P(fail | previous execution failed) vs P(fail | previous succeeded), with Fisher's exact test.
2. **Galton-Watson delegation** (paper Section 5.1 and 7) - the Magentic-One orchestrator's
   delegations form a tree; the paper's estimator (:func:`agentq.calibration.estimate_delegation`)
   is applied to the real offspring counts and the implied Borel tree-size law is compared with
   the real tree sizes.
3. **Failed tasks stop at the failing step** - the position of the annotated decisive error,
   and the share of a failed run's agent turns that come after it.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats

from agentq.branching import borel_tail
from agentq.calibration import estimate_delegation
from agentq.datasets.whowhen import load_whowhen
from agentq.experiments.paper import Profiles

TOOL_SPEAKERS = {"Computer_terminal", "human"}


def _wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    z = 1.959964
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def _rate(xs: list[bool]) -> dict[str, Any]:
    k, n = int(sum(xs)), len(xs)
    lo, hi = _wilson(k, n)
    return {"rate": k / n if n else None, "n": n, "ci95": [lo, hi]}


def _tool_failures(runs: list[Any]) -> dict[str, Any]:
    seqs = [r.tool_outcomes() for r in runs]
    allx = [x for s in seqs for x in s]
    after_fail = [b for s in seqs for a, b in zip(s[:-1], s[1:], strict=True) if a]
    after_ok = [b for s in seqs for a, b in zip(s[:-1], s[1:], strict=True) if not a]
    table = [[sum(after_fail), len(after_fail) - sum(after_fail)], [sum(after_ok), len(after_ok) - sum(after_ok)]]
    odds, pval = stats.fisher_exact(table)
    return {
        "executions": len(allx),
        "runs_with_executions": sum(1 for s in seqs if s),
        "failure": _rate(allx),
        "fail_after_fail": _rate(after_fail),
        "fail_after_success": _rate(after_ok),
        "fisher_odds_ratio": float(odds),
        "fisher_pvalue": float(pval),
    }


def _borel_quantile(m: float, q: float) -> int:
    n = 1
    while borel_tail(n, m) > 1 - q and n < 100_000:
        n += 1
    return n


def _delegation(runs: list[Any]) -> dict[str, Any]:
    n_deleg = np.array([len(r.delegations()) for r in runs], dtype=float)
    sizes = 1.0 + n_deleg  # orchestrator + one sub-agent invocation per delegation
    # offspring of every agent invocation: the orchestrator has N children, sub-agents have none
    offspring = [int(x) for x in n_deleg] + [0] * int(n_deleg.sum())
    est = estimate_delegation(offspring)
    m = est["m_hat"]
    targets: dict[str, int] = {}
    for r in runs:
        for t in r.delegations():
            targets[t] = targets.get(t, 0) + 1
    turns = [t for r in runs for t in r.subagent_turns_per_delegation()]
    return {
        "runs": len(runs),
        "delegations_per_task": {
            "mean": float(n_deleg.mean()),
            "var": float(n_deleg.var(ddof=1)),
            "dispersion_index": float(n_deleg.var(ddof=1) / n_deleg.mean()),
            "min": int(n_deleg.min()),
            "max": int(n_deleg.max()),
        },
        "targets": dict(sorted(targets.items(), key=lambda kv: -kv[1])),
        "subagent_turns_per_delegation": {str(k): turns.count(k) for k in sorted(set(turns))},
        "max_depth": 1,
        "tree_size": {
            "mean": float(sizes.mean()),
            "var": float(sizes.var(ddof=1)),
            "p50": float(np.percentile(sizes, 50)),
            "p99": float(np.percentile(sizes, 99)),
        },
        "galton_watson_estimate": est,
        "galton_watson_implied_tree_size": {
            "mean": 1.0 / (1.0 - m),
            "var": m / (1.0 - m) ** 3,  # Poisson offspring: sigma^2 = m (Eq. (2))
            "p50": _borel_quantile(m, 0.5),
            "p99": _borel_quantile(m, 0.99),
            "p_size_1": math.exp(-m),
        },
        "real_p_size_1": float((sizes == 1).mean()),
    }


def _mistakes(runs: list[Any]) -> dict[str, Any]:
    pos, after, agent_turns = [], [], []
    agents: dict[str, int] = {}
    for r in runs:
        n = len(r.history)
        turns = [i for i in range(n) if r.speaker(i) not in TOOL_SPEAKERS]
        agent_turns.append(len(turns))
        pos.append(r.mistake_step / max(1, n - 1))
        after.append(sum(1 for i in turns if i > r.mistake_step) / max(1, len(turns)))
        agents[r.mistake_agent] = agents.get(r.mistake_agent, 0) + 1
    pos_a, after_a, turns_a = np.array(pos), np.array(after), np.array(agent_turns, dtype=float)
    return {
        "decisive_error_position": {
            q: float(np.percentile(pos_a, p)) for q, p in (("p25", 25), ("p50", 50), ("p75", 75))
        },
        "fraction_of_agent_turns_after_error": {"mean": float(after_a.mean()), "p50": float(np.median(after_a))},
        "agent_turns_per_run": {
            "mean": float(turns_a.mean()),
            "var": float(turns_a.var(ddof=1)),
            "max": int(turns_a.max()),
        },
        "mistake_agents": dict(sorted(agents.items(), key=lambda kv: -kv[1])),
    }


def real_multiagent(profiles: Profiles, params: dict[str, Any]) -> dict[str, Any]:
    """Failure clustering, delegation structure and decisive-error position in Who&When."""
    data = load_whowhen(params["whowhen_dir"])
    alg, hand = data["Algorithm-Generated"], data["Hand-Crafted"]
    return {
        "n_runs": {k: len(v) for k, v in data.items()},
        "all_runs_failed": True,
        "tool_failures_captainagent": _tool_failures(alg),
        "delegation_magentic_one": _delegation(hand),
        "decisive_errors": {"Algorithm-Generated": _mistakes(alg), "Hand-Crafted": _mistakes(hand)},
        "notes": [
            "Algorithm-Generated histories are truncated at 10 messages; run lengths there are not natural.",
            "All runs failed the task, so success-conditioned quantities are not identifiable.",
        ],
    }


MULTIAGENT_EXPERIMENTS = {"real_multiagent": real_multiagent}
