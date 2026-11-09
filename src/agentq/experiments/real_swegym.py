"""SWE-Gym (ICML 2025): success, failure modes and cost per success of real coding agents.

The first dataset with *both* outcomes from one system, so the model's success and cost results
can be tested directly. Per group of runs (all, per model, per model and iteration cap):

* outcome mix - resolved, **detected failure** (no patch produced: aborted by the stuck-loop
  detector or gave up), **silent failure** (a patch that fails the hidden tests);
* how runs end (``finish`` / ``cap`` / ``abort``) and the repeated-action tail of aborts;
* LLM calls per run by outcome and the **measured cost per success** (all calls / resolved runs),
  against the model's ``P_s`` and ``C_s = n_bar / (1 - p)`` (Eq. (8)) calibrated on the resolved runs;
* step counts of resolved runs vs geometric, the per-attempt fan-out law, failure clustering
  (strict and broad definitions) and context growth.

Latency is tested separately by ``real_validation_swegym`` (trace replay through the pools).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from scipy import stats

from agentq.config import WorkflowConfig
from agentq.datasets.swegym import extract_swegym, load_records
from agentq.experiments.paper import Profiles
from agentq.experiments.real import _step_diagnostics
from agentq.experiments.real_multiagent import _rate
from agentq.structure import structural_quantities


def _clustering(records: list[dict[str, Any]], col: int) -> dict[str, Any]:
    seqs = [[bool(a[col]) for a in r["attempts"]] for r in records]
    after_fail = [b for s in seqs for a, b in zip(s[:-1], s[1:], strict=True) if a]
    after_ok = [b for s in seqs for a, b in zip(s[:-1], s[1:], strict=True) if not a]
    table = [[sum(after_fail), len(after_fail) - sum(after_fail)], [sum(after_ok), len(after_ok) - sum(after_ok)]]
    odds, pval = stats.fisher_exact(table)
    return {
        "failure": _rate([x for s in seqs for x in s]),
        "fail_after_fail": _rate(after_fail),
        "fail_after_success": _rate(after_ok),
        "odds_ratio": float(odds),
        "fisher_pvalue": float(pval),
    }


def _steps_and_retries(rec: dict[str, Any]) -> tuple[int, int]:
    """(successful steps, longest run of consecutive strict failures)."""
    steps = longest = cur = 0
    for a in rec["attempts"]:
        if a[3]:
            cur += 1
            longest = max(longest, cur)
        else:
            steps += 1
            cur = 0
    return steps, longest


def _dist(x: np.ndarray) -> dict[str, float]:
    if x.size == 0:
        return {}
    return {
        "mean": float(x.mean()),
        "p50": float(np.median(x)),
        "p90": float(np.percentile(x, 90)),
        "max": float(x.max()),
    }


def _group(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    res = [r for r in records if r["resolved"]]
    silent = [r for r in records if not r["resolved"] and not r["empty_patch"]]
    detected = [r for r in records if not r["resolved"] and r["empty_patch"]]
    calls = {
        k: np.array([len(r["attempts"]) for r in g], float)
        for k, g in (("resolved", res), ("silent_failure", silent), ("detected_failure", detected), ("all", records))
    }
    total_calls = float(calls["all"].sum())
    out: dict[str, Any] = {
        "runs": n,
        "resolved": _rate([r["resolved"] for r in records]),
        "outcome_mix": {
            "resolved": len(res) / n,
            "silent_failure": len(silent) / n,
            "detected_failure": len(detected) / n,
        },
        "ending_by_outcome": {
            k: dict(Counter(r["ending"] for r in g))
            for k, g in (("resolved", res), ("silent_failure", silent), ("detected_failure", detected))
        },
        "abort_tail_repeats": _dist(np.array([r["tail_repeats"] for r in records if r["ending"] == "abort"], float)),
        "llm_calls_per_run": {k: _dist(v) for k, v in calls.items()},
        "share_of_calls": {k: float(v.sum() / total_calls) for k, v in calls.items() if k != "all"},
        "measured_calls_per_success": total_calls / len(res) if res else None,
    }
    if len(res) >= 10:
        sr = [_steps_and_retries(r) for r in res]
        steps = np.array([s for s, _ in sr], float)
        att = [a for r in res for a in r["attempts"]]
        p = float(np.mean([a[3] for a in att]))
        r_max = int(max(m for _, m in sr))
        wf = WorkflowConfig(
            mean_steps=max(1.0, float(steps.mean())),
            fanout_kind="poisson",
            fanout=1.0,
            failure_prob=min(p, 0.99),
            max_retries=min(r_max, 100),
        )
        sq = structural_quantities(wf)
        meas = out["measured_calls_per_success"]
        out["model_calibrated_on_resolved"] = {
            "p_strict": p,
            "R_max_observed": r_max,
            "n_bar": float(steps.mean()),
            "model_success_prob": sq.success_prob,
            "model_calls_per_success": sq.calls_per_success,
            "measured_success_prob": len(res) / n,
            "measured_calls_per_success": meas,
            "cost_understatement_factor": meas / sq.calls_per_success,
        }
        out["resolved_steps"] = _step_diagnostics(steps, max(1.0, float(steps.mean())))
        out["resolved_hit_cap"] = float(np.mean([r["ending"] == "cap" for r in res]))
    fan = np.array([a[2] for r in records for a in r["attempts"]], float)
    pmf = Counter(int(k) for k in fan)
    out["fanout"] = {
        "pmf": {str(k): v / fan.size for k, v in sorted(pmf.items())},
        "mean": float(fan.mean()),
        "dispersion_index": float(fan.var() / fan.mean()),
        "e_k2_over_e_k": float((fan**2).mean() / fan.mean()),
    }
    out["failures_strict"] = _clustering(records, 3)
    out["failures_broad"] = _clustering(records, 4)
    ctx = np.array([a[0] for r in records for a in r["attempts"]], float)
    gen = np.array([a[1] for r in records for a in r["attempts"]], float)
    first = np.array([r["attempts"][0][0] for r in records], float)
    last = np.array([r["attempts"][-1][0] for r in records], float)
    out["tokens"] = {
        "context": {"p50": float(np.median(ctx)), "p99": float(np.percentile(ctx, 99))},
        "output": {"p50": float(np.median(gen)), "p99": float(np.percentile(gen, 99))},
        "context_first_call_mean": float(first.mean()),
        "context_last_call_mean": float(last.mean()),
    }
    return out


def real_swegym(profiles: Profiles, params: dict[str, Any]) -> dict[str, Any]:
    cache = extract_swegym(params["swegym_dir"], params["swegym_records"])
    recs = [r for r in load_records(cache) if r["attempts"]]
    groups: dict[str, list[dict[str, Any]]] = {"all": recs}
    for r in recs:
        groups.setdefault(r["model"], []).append(r)
        groups.setdefault(f"{r['model']}|maxiter={r['maxiter']}", []).append(r)
    return {
        "n_runs": len(recs),
        "n_instances": len({r["instance_id"] for r in recs}),
        "run_configs": dict(Counter(r["run_id"] for r in recs)),
        "groups": {k: _group(v) for k, v in sorted(groups.items())},
    }


SWEGYM_EXPERIMENTS = {"real_swegym": real_swegym}
