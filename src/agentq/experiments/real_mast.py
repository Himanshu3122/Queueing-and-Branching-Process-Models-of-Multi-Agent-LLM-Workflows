"""MAST (NeurIPS 2025): measured end-to-end latency of real multi-agent runs, and failure modes.

For the frameworks that log timestamps (ChatDev, MetaGPT, OpenManus) every run's steps were
executed one at a time against a hosted LLM, so the measured task latency is a **zero-load** test
of the model's latency composition (Eqs. (16)-(19) without queueing). Per framework and LLM:

* ``measured`` - distribution over runs of the summed step durations;
* ``model`` - the paper's model calibrated on those runs: geometric step count with the observed
  mean, i.i.d. lognormal step time with the observed mean and CV, no failures, no tool fork,
  no queueing (``ModelOptions(use_queueing=False)``);
* ``empirical_steps_iid`` - Monte Carlo with the *observed* step-count distribution and i.i.d.
  draws from the observed step times. ``model -> empirical_steps_iid`` isolates the step-count law,
  ``empirical_steps_iid -> measured`` isolates the i.i.d. step-time assumption.

Also: step-time dependence within a run (variance of run-mean step time vs the i.i.d. value),
measured latency vs tokens (ChatDev only: its log brackets each LLM request exactly), and MAST
failure-mode prevalence (all 1,642 runs).
OpenManus runs are censored by a 900 s time limit and are excluded from the latency test.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from agentq.calibration import fit_service_times
from agentq.config import PoolConfig, SystemConfig, WorkflowConfig
from agentq.datasets.mast import CATEGORIES, MODES, TimedTrace, load_mast, timed_traces
from agentq.experiments.common import relative_error
from agentq.experiments.paper import Profiles, _options
from agentq.experiments.real import _step_diagnostics
from agentq.model import predict

Q = {"mean": None, "p50": 50, "p95": 95, "p99": 99}


def _quantiles(x: np.ndarray) -> dict[str, float]:
    return {k: float(x.mean() if p is None else np.percentile(x, p)) for k, p in Q.items()}


def _within_run_dependence(group: list[TimedTrace]) -> dict[str, float]:
    """Observed variance of run-mean step time / the variance i.i.d. steps would give."""
    means = np.array([np.mean(t.step_s) for t in group])
    counts = np.array([len(t.step_s) for t in group], dtype=float)
    pooled = np.concatenate([t.step_s for t in group])
    expected = pooled.var() * np.mean(1.0 / counts)
    return {"variance_ratio": float(means.var() / expected) if expected > 0 else float("nan")}


def _latency_test(group: list[TimedTrace], params: dict[str, Any], seed: int) -> dict[str, Any]:
    steps = np.array([len(t.step_s) for t in group], dtype=float)
    pooled = np.concatenate([t.step_s for t in group])
    measured = np.array([sum(t.step_s) for t in group])
    mean_s = float(pooled.mean())
    cv = float(max(pooled.std() / mean_s, 0.05))
    system = SystemConfig(
        name="mast",
        workflow=WorkflowConfig(
            mean_steps=float(steps.mean()), fanout_kind="fixed", fanout=0, failure_prob=0.0, max_retries=0
        ),
        llm=PoolConfig(servers=1, mean_service=max(mean_s, 1e-3), cv=cv),
        tool=PoolConfig(servers=1, mean_service=1.0, cv=1.0),
    )
    pred = predict(system, 1e-9, _options(params, use_queueing=False)).to_dict()
    counts = np.bincount(steps.astype(int))[1:]
    emp = system.with_updates(**{"workflow.step_pmf": tuple(float(c) for c in counts / counts.sum())})
    pred_emp = predict(emp, 1e-9, _options(params, use_queueing=False)).to_dict()
    rng = np.random.default_rng(seed)
    n_mc = params.get("mc_runs", 200_000)
    n_draw = rng.choice(steps.astype(int), n_mc)
    mc = np.array([pooled[rng.integers(0, pooled.size, k)].sum() for k in n_draw])
    meas_q, mc_q = _quantiles(measured), _quantiles(mc)
    model_q = {k: pred[k] for k in Q}
    model_emp_q = {k: pred_emp[k] for k in Q}
    return {
        "runs": len(group),
        "steps": _step_diagnostics(steps, float(steps.mean())),
        "step_time": {
            "mean": mean_s,
            "cv": float(pooled.std() / mean_s),
            "aic_best": fit_service_times(pooled[pooled > 0])[0].family if (pooled > 0).sum() >= 10 else None,
        },
        "within_run_dependence": _within_run_dependence(group),
        "share_of_wall_time_in_steps": float(np.median(measured / np.maximum([t.total_s for t in group], 1e-9))),
        "measured": meas_q,
        "model": model_q,
        "empirical_steps_iid": mc_q,
        "model_empirical_steps": model_emp_q,
        "error_model_vs_measured": {k: relative_error(model_q[k], meas_q[k]) for k in Q},
        "error_model_empirical_steps_vs_measured": {k: relative_error(model_emp_q[k], meas_q[k]) for k in Q},
        "error_empirical_steps_iid_vs_measured": {k: relative_error(mc_q[k], meas_q[k]) for k in Q},
    }


def _token_fit(group: list[TimedTrace]) -> dict[str, Any] | None:
    rows = [(s, p, c) for t in group for s, (p, c) in zip(t.step_s, t.tokens, strict=False) if p >= 0]
    if len(rows) < 30:
        return None
    y = np.array([r[0] for r in rows], dtype=float)
    x = np.c_[np.ones(len(rows)), [r[1] for r in rows], [r[2] for r in rows]]
    b = np.linalg.lstsq(x, y, rcond=None)[0]
    comp = np.array([r[2] for r in rows], dtype=float)
    long = comp >= 50
    return {
        "calls": len(rows),
        "overhead_s": float(b[0]),
        "prefill_ms_per_token": float(1000 * b[1]),
        "decode_ms_per_token": float(1000 * b[2]),
        "r2": float(1 - np.var(y - x @ b) / np.var(y)),
        "median_ms_per_completion_token": float(1000 * np.median(y[long] / comp[long])) if long.any() else None,
    }


def _failure_modes(records: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by[r["mas_name"]].append(r)
    by["all"] = list(records)
    out = {}
    for mas, rs in by.items():
        ann = [r["mast_annotation"] for r in rs]
        mode = {m: float(np.mean([a[m] for a in ann if a.get(m) is not None] or [np.nan])) for m in MODES}
        cat = {c: float(np.mean([any(a.get(m) == 1 for m in ms) for a in ann])) for c, ms in CATEGORIES.items()}
        out[mas] = {
            "runs": len(rs),
            "any_failure_mode": float(np.mean([any(v == 1 for v in a.values()) for a in ann])),
            "by_category": cat,
            "by_mode": mode,
        }
    return out


def real_mast(profiles: Profiles, params: dict[str, Any]) -> dict[str, Any]:
    records = load_mast(params["mast_file"])
    traces = timed_traces(records)
    groups: dict[str, list[TimedTrace]] = defaultdict(list)
    for t in traces:
        groups[f"{t.mas}/{t.llm}"].append(t)
    latency, tokens, http = {}, {}, {}
    for i, (name, g) in enumerate(sorted(groups.items())):
        status: dict[str, int] = {}
        for t in g:
            for k, v in t.http_status.items():
                status[k] = status.get(k, 0) + v
        if status:
            http[name] = status
        # only ChatDev logs both ends of each LLM request; OpenManus call starts include tool work
        fit = _token_fit(g) if g[0].mas == "ChatDev" else None
        if fit:
            tokens[name] = fit
        if g[0].mas in params["latency_frameworks"] and len(g) >= params.get("min_runs", 30):
            latency[name] = _latency_test(g, params, seed=i)
    return {
        "n_runs": len(records),
        "timed_runs": {k: len(v) for k, v in sorted(groups.items())},
        "zero_load_latency": latency,
        "latency_vs_tokens": tokens,
        "http_status": http,
        "failure_modes": _failure_modes(records),
    }


MAST_EXPERIMENTS = {"real_mast": real_mast}
