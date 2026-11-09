"""Experiments on real, public datasets (not part of the paper's synthetic protocol).

* ``real_calibration`` - estimate the model's parameters from AgentInstruct trajectories and
  test its structural assumptions (geometric step counts, independent failures, service-time
  family, Poisson arrivals) against AgentInstruct and the Azure LLM inference trace.
* ``real_validation`` - compare the calibrated model with (a) the reference simulator under the
  model's own assumptions, (b) a replay of the real trajectories with Poisson arrivals and (c) the
  same replay with the real Azure arrival timestamps. The gap (a) is approximation error, (a)->(b)
  is workflow-structure mismatch, (b)->(c) is arrival burstiness.

What is measured vs assumed is listed in ``docs/real_data.md``. LLM service times derive from real
token counts through an assumed latency model; tool service times are assumed (not recorded).
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from agentq.calibration import calibrate, fit_service_times
from agentq.config import PoolConfig, SystemConfig, WorkflowConfig
from agentq.datasets.agentinstruct import Trajectory, load_agentinstruct, to_attempt_rows
from agentq.datasets.azure import arrival_statistics, load_azure_arrivals, rate_surges, rescale_arrivals
from agentq.datasets.mooncake import load_mooncake, prefix_linkage, timestamp_resolution
from agentq.datasets.service import TokenLatencyModel
from agentq.experiments.common import aggregate, relative_error
from agentq.experiments.paper import LAT, Profiles, _options, _pred_dict
from agentq.sim import ReplayTask, poisson_arrivals, simulate_chain, simulate_replay
from agentq.structure import rate_for_llm_utilization

SIM_KEYS = [*LAT, "success_prob", "llm_calls_per_task", "rho_llm", "rho_tool"]
MIXED = "mixed"
log = logging.getLogger(__name__)


def load_arrival_trace(spec: dict[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Arrival times and token columns of a real trace: ``{"kind": "azure" | "mooncake", "path": ...}``."""
    if spec["kind"] == "azure":
        return load_azure_arrivals(spec["path"])
    if spec["kind"] == "mooncake":
        return load_mooncake(spec["path"])
    raise ValueError(f"unknown arrival trace kind '{spec['kind']}'")


def _arrival_specs(params: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Named arrival traces; defaults to the Azure trace (reference key ``replay_azure``)."""
    return params.get("arrivals") or {"azure": {"kind": "azure", "path": params["azure_csv"]}}


# --------------------------------------------------------------------------- data access
@lru_cache(maxsize=2)
def _trajectories(data_dir: str) -> dict[str, list[Trajectory]]:
    return load_agentinstruct(data_dir)


def _workloads(params: dict[str, Any]) -> dict[str, list[Trajectory]]:
    if "swegym_records" in params:  # SWE-Gym (ICML 2025) workloads, keyed by model
        from agentq.datasets.swegym import swegym_workloads

        return swegym_workloads(params["swegym_records"], params["envs"])
    data = _trajectories(str(Path(params["agentinstruct_dir"])))
    envs = params["envs"]
    out = {e: data[e] for e in envs if e != MIXED}
    if MIXED in envs:
        out[MIXED] = [t for e in data for t in data[e]]
    return out


def _latency(params: dict[str, Any], override: dict[str, float] | None = None) -> TokenLatencyModel:
    return TokenLatencyModel(**{**params.get("latency", {}), **(override or {})})


def _service_times(trajs: list[Trajectory], latency: TokenLatencyModel) -> np.ndarray:
    return np.array([latency.service_time(a.context_tokens, a.output_tokens) for t in trajs for a in t.attempts])


def calibrated_system(
    trajs: list[Trajectory], latency: TokenLatencyModel, params: dict[str, Any], name: str
) -> tuple[SystemConfig, dict[str, Any]]:
    """Fit the model to trajectories with :func:`agentq.calibration.calibrate`.

    The LLM pool uses the empirical mean and CV of the token-derived service times; the tool pool
    is the assumed configuration in ``params["tool"]``; ``k_L`` is ``params["llm_servers"]``.
    """
    cal = calibrate(to_attempt_rows(trajs, latency))
    s = _service_times(trajs, latency)
    fan = round(cal.fanout_mean) if cal.fanout_kind == "fixed" else cal.fanout_mean
    system = SystemConfig(
        name=name,
        workflow=WorkflowConfig(
            mean_steps=max(1.0, cal.mean_steps),
            fanout_kind=cal.fanout_kind,  # type: ignore[arg-type]
            fanout=fan,
            failure_prob=cal.failure_prob,
            max_retries=cal.max_retries_observed,
        ),
        llm=PoolConfig(servers=params["llm_servers"], mean_service=float(s.mean()), cv=float(s.std() / s.mean())),
        tool=PoolConfig(**params["tool"]),
    )
    if params.get("step_law", "geometric") == "empirical":
        system = system.with_updates(**{"workflow.step_pmf": empirical_step_pmf(trajs)})
    return system, cal.to_dict()


def empirical_step_pmf(trajs: list[Trajectory]) -> tuple[float, ...]:
    """Observed step-count law ``P(N = n)``, n = 1..max (runs with no successful step count as 1)."""
    counts = np.bincount(np.array([max(1, t.steps) for t in trajs]))[1:]
    return tuple(float(c) for c in counts / counts.sum())


def _replay_tasks(trajs: list[Trajectory], latency: TokenLatencyModel) -> list[ReplayTask]:
    return [
        (
            tuple((latency.service_time(a.context_tokens, a.output_tokens), a.n_tool_calls) for a in t.attempts),
            t.success,
        )
        for t in trajs
    ]


# --------------------------------------------------------------------------- diagnostics
def _step_diagnostics(steps: np.ndarray, mean_steps: float) -> dict[str, float]:
    """Compare the empirical step-count law with the geometric law the model assumes."""
    q = 1.0 / mean_steps
    support = np.arange(1, int(steps.max()) + 1)
    emp_cdf = np.array([(steps <= k).mean() for k in support])
    geo_cdf = 1.0 - (1.0 - q) ** support
    geo_var = (1.0 - q) / q**2
    return {
        "mean": float(steps.mean()),
        "var": float(steps.var(ddof=1)),
        "geometric_var": float(geo_var),
        "var_ratio_to_geometric": float(steps.var(ddof=1) / geo_var) if geo_var > 0 else None,
        "p99": float(np.percentile(steps, 99)),
        "geometric_p99": float(math.ceil(math.log(0.01) / math.log(1.0 - q))) if q < 1 else 1.0,
        "ks_distance_to_geometric": float(np.max(np.abs(emp_cdf - geo_cdf))),
        "min": int(steps.min()),
        "max": int(steps.max()),
    }


def _failure_diagnostics(trajs: list[Trajectory]) -> dict[str, float | None]:
    """Is a failure more likely right after a failure? (The model assumes independent attempts.)"""
    after_fail, after_ok = [], []
    for t in trajs:
        for prev, cur in zip(t.attempts[:-1], t.attempts[1:], strict=True):
            (after_fail if prev.failed else after_ok).append(cur.failed)
    p_ff = float(np.mean(after_fail)) if after_fail else None
    p_fo = float(np.mean(after_ok)) if after_ok else None
    tasks_with_fail = float(np.mean([any(a.failed for a in t.attempts) for t in trajs]))
    return {
        "p_fail_after_fail": p_ff,
        "n_after_fail": len(after_fail),
        "p_fail_after_success": p_fo,
        "fraction_tasks_with_a_failure": tasks_with_fail,
    }


def _service_diagnostics(trajs: list[Trajectory], latency: TokenLatencyModel) -> dict[str, Any]:
    """Service-time family (AIC), and dependence on position in the task (context growth)."""
    s = _service_times(trajs, latency)
    pos = np.array([i for t in trajs for i in range(t.n_attempts)])
    first = np.array([latency.service_time(t.attempts[0].context_tokens, t.attempts[0].output_tokens) for t in trajs])
    last = np.array([latency.service_time(t.attempts[-1].context_tokens, t.attempts[-1].output_tokens) for t in trajs])
    ctx = np.array([a.context_tokens for t in trajs for a in t.attempts])
    out = np.array([a.output_tokens for t in trajs for a in t.attempts])
    rho, pval = stats.spearmanr(pos, s)
    return {
        "mean": float(s.mean()),
        "cv": float(s.std() / s.mean()),
        "p99": float(np.percentile(s, 99)),
        "aic_fits": [f.to_dict() for f in fit_service_times(s)],
        "spearman_position_vs_service": float(rho),
        "spearman_pvalue": float(pval),
        "mean_first_call": float(first.mean()),
        "mean_last_call": float(last.mean()),
        "context_tokens": {"median": float(np.median(ctx)), "p99": float(np.percentile(ctx, 99))},
        "output_tokens": {"median": float(np.median(out)), "p99": float(np.percentile(out, 99))},
    }


def real_calibration(profiles: Profiles, params: dict[str, Any]) -> dict[str, Any]:
    """Calibrate per environment and test the model's assumptions on real data."""
    latency = _latency(params)
    envs: dict[str, Any] = {}
    for name, trajs in _workloads(params).items():
        system, cal = calibrated_system(trajs, latency, params, name)
        steps = np.array([t.steps for t in trajs], dtype=float)
        fan = np.array([a.n_tool_calls for t in trajs for a in t.attempts], dtype=float)
        envs[name] = {
            "n_trajectories": len(trajs),
            "calibration": cal,
            "calibrated_system": system.model_dump(),
            "steps": _step_diagnostics(steps, system.workflow.mean_steps),
            "attempts_per_task": float(np.mean([t.n_attempts for t in trajs])),
            "failures": _failure_diagnostics(trajs),
            "fanout": {
                "mean": float(fan.mean()),
                "zero_fraction": float((fan == 0).mean()),
                "model_kind": system.workflow.fanout_kind,
                "model_fanout": system.workflow.fanout,
            },
            "llm_service": _service_diagnostics(trajs, latency),
        }
    times, tokens = load_azure_arrivals(params["azure_csv"])
    azure = arrival_statistics(times)
    azure["tokens"] = {k: {"mean": float(v.mean()), "cv": float(v.std() / v.mean())} for k, v in tokens.items()}
    rng = np.random.default_rng(params.get("seed", 0))
    azure["poisson_reference"] = arrival_statistics(poisson_arrivals(rng, azure["mean_rate"], len(times)))
    return {"latency_model": latency.to_dict(), "environments": envs, "azure_arrivals": azure}


# --------------------------------------------------------------------------- validation
def _replay(
    tasks: list[ReplayTask], arrival_fn: Any, system: SystemConfig, params: dict[str, Any]
) -> dict[str, tuple[float, float]]:
    runs = []
    for seed in range(params["seeds"]):
        rng = np.random.default_rng(10_000 + seed)
        arr = arrival_fn(seed, rng)
        sample = [tasks[i] for i in rng.integers(0, len(tasks), arr.size)]
        runs.append(simulate_replay(sample, arr, system.llm.servers, system.tool, seed=seed))
    return aggregate(runs, SIM_KEYS, params.get("spread", "ci"))


def _validate_point(
    system: SystemConfig,
    tasks: list[ReplayTask],
    traces: dict[str, np.ndarray],
    rho: float,
    params: dict[str, Any],
) -> dict[str, Any]:
    opts = _options(params)
    rate = rate_for_llm_utilization(system, rho)
    n = params["n_tasks"]
    model = _pred_dict(system, rate, opts)
    refs: dict[str, dict[str, tuple[float, float]]] = {}
    if params.get("model_sim", True):
        runs = [simulate_chain(system, rate, n, seed=s) for s in range(params["seeds"])]
        refs["sim_model_assumptions"] = aggregate(runs, SIM_KEYS, params.get("spread", "ci"))
    refs["replay_poisson"] = _replay(tasks, lambda s, rng: poisson_arrivals(rng, rate, n), system, params)
    for name, times in traces.items():
        refs[f"replay_{name}"] = _replay(
            tasks,
            lambda s, rng, times=times: rescale_arrivals(times, rate, n=n, offset_fraction=s / params["seeds"]),
            system,
            params,
        )
    return {
        "rho_llm": rho,
        "rate": rate,
        "model": model,
        "references": refs,
        "error": {
            ref: {k: relative_error(model.get(k), v[k][0]) for k in ("mean", "p50", "p95", "p99")}
            for ref, v in refs.items()
        },
    }


def real_validation(profiles: Profiles, params: dict[str, Any]) -> dict[str, Any]:
    """Model vs trace replay with Poisson and with real (Azure) arrivals, per environment and load."""
    latency = _latency(params)
    specs = _arrival_specs(params)
    traces = {name: load_arrival_trace(spec)[0] for name, spec in specs.items()}
    out: dict[str, Any] = {"latency_model": latency.to_dict(), "arrival_traces": specs, "environments": {}}
    workloads = _workloads(params)
    total, done = len(workloads) * len(params["rho_llm"]), 0
    for name, trajs in workloads.items():
        system, _ = calibrated_system(trajs, latency, params, name)
        tasks = _replay_tasks(trajs, latency)
        rows = []
        for rho in params["rho_llm"]:
            rows.append(_validate_point(system, tasks, traces, rho, params))
            done += 1
            log.info("progress %d/%d: workload=%s rho_L=%.2f done", done, total, name, rho)
        out["environments"][name] = {"calibrated_system": system.model_dump(), "rows": rows}
    sens = params.get("sensitivity")
    if sens:
        rows = []
        for label, override in sens["latency"].items():
            lat = _latency(params, override)
            for name in sens["envs"]:
                system, _ = calibrated_system(workloads[name], lat, params, name)
                point = _validate_point(system, _replay_tasks(workloads[name], lat), traces, sens["rho_llm"], params)
                rows.append({"latency": label, "latency_model": lat.to_dict(), "env": name, **point})
        out["sensitivity"] = rows
    return out


def real_arrival_traces(profiles: Profiles, params: dict[str, Any]) -> dict[str, Any]:
    """Characterize real arrival traces: burstiness, rate surges, token lengths, implied LLM service.

    For Mooncake traces also: timestamp resolution and whether sessions can be rebuilt from
    prefix hashes. The implied service time uses the same assumed token latency model as the
    AgentInstruct experiments, so the token distributions can be compared directly.
    """
    latency = _latency(params)
    rng = np.random.default_rng(params.get("seed", 0))
    out: dict[str, Any] = {"latency_model": latency.to_dict(), "traces": {}}
    for name, spec in _arrival_specs(params).items():
        times, tokens = load_arrival_trace(spec)
        ctx = tokens["input_tokens"] if "input_tokens" in tokens else tokens["context_tokens"]
        gen = tokens["output_tokens"] if "output_tokens" in tokens else tokens["generated_tokens"]
        service = 1e-3 * (latency.prefill_ms_per_token * ctx + latency.decode_ms_per_token * gen) + latency.overhead_s
        rate = (times.size - 1) / times[-1]
        entry: dict[str, Any] = {
            "spec": spec,
            "arrivals": arrival_statistics(times),
            "rate_surges": rate_surges(times),
            "poisson_reference": arrival_statistics(poisson_arrivals(rng, rate, times.size)),
            "input_tokens": _summary(ctx),
            "output_tokens": _summary(gen),
            "implied_llm_service": {**_summary(service), "aic_fits": [f.to_dict() for f in fit_service_times(service)]},
        }
        if spec["kind"] == "mooncake":
            entry["timestamp_resolution"] = timestamp_resolution(spec["path"])
            entry["session_linkage"] = prefix_linkage(spec["path"])
        out["traces"][name] = entry
        log.info("characterized arrival trace %s (%d requests)", name, times.size)
    return out


def _summary(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    return {
        "mean": float(x.mean()),
        "cv": float(x.std() / x.mean()),
        "p50": float(np.median(x)),
        "p99": float(np.percentile(x, 99)),
    }


REAL_EXPERIMENTS = {
    "real_calibration": real_calibration,
    "real_validation": real_validation,
    "real_arrival_traces": real_arrival_traces,
    "real_validation_mooncake": real_validation,
    "real_validation_swegym": real_validation,
    "real_validation_empirical": real_validation,
    "real_validation_swegym_empirical": real_validation,
}
