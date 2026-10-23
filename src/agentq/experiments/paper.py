"""Experiments of the paper (journal version, Section 7-8), one function per table/figure.

Each function takes the loaded profiles and a parameter dict (from ``configs/experiments.yaml``)
and returns a JSON-serializable dict. The ``full`` parameter sets reproduce the paper's protocol;
``smoke`` sets are tiny and only check that the pipeline runs.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from agentq.branching import (
    borel_tail,
    criticality_relative_se,
    expected_total_progeny,
    poisson_extinction_probability,
    simulate_total_progeny,
    variance_total_progeny,
)
from agentq.bursts import MMPP2, interarrival_scv, quasi_static_prediction
from agentq.config import SystemConfig
from agentq.experiments.common import aggregate, mean_ci, relative_error
from agentq.grid import Grid
from agentq.model import ModelOptions, mean_value_baseline, predict
from agentq.planning import max_rate_for_p99
from agentq.sim import mmpp_arrivals, simulate_chain, simulate_tree
from agentq.structure import rate_for_llm_utilization, stability_limit, structural_quantities
from agentq.trees import agents_per_tree, predict_tree

Profiles = dict[str, SystemConfig]
LAT = ("mean", "p50", "p95", "p99")


def _options(params: dict[str, Any], **kw: Any) -> ModelOptions:
    g = params.get("grid", {})
    return ModelOptions(grid=Grid(dt=g.get("dt", 0.02), n=g.get("n", 30_000)), **kw)


def _pred_dict(system: SystemConfig, rate: float, opts: ModelOptions) -> dict[str, Any]:
    return predict(system, rate, opts).to_dict()


def _sim_chain(
    system: SystemConfig, rate: float, params: dict[str, Any], spread: str, arrivals: MMPP2 | None = None
) -> dict[str, tuple[float, float]]:
    runs = [simulate_chain(system, rate, params["n_tasks"], seed=s, arrivals=arrivals) for s in range(params["seeds"])]
    return aggregate(runs, [*LAT, "success_prob", "llm_calls_per_task", "rho_llm", "rho_tool"], spread)


def _errors(model: dict[str, Any], sim: dict[str, tuple[float, float]]) -> dict[str, float | None]:
    return {k: relative_error(model.get(k), sim[k][0]) for k in ("mean", "p95", "p99")}


# --------------------------------------------------------------------------- Table 3, Fig. 4
def branching(profiles: Profiles, params: dict[str, Any]) -> dict[str, Any]:
    """Total progeny vs Eqs. (1)-(4), supercritical extinction, overdispersion, criticality estimation."""
    rng = np.random.default_rng(params.get("seed", 1))
    progeny = []
    for m in params["progeny_m"]:
        z = simulate_total_progeny(rng, m, params["n_trees"])
        progeny.append(
            {
                "m": m,
                "EZ_eq": expected_total_progeny(m),
                "EZ_sim": float(z.mean()),
                "sd_eq": variance_total_progeny(m, m) ** 0.5,
                "sd_sim": float(z.std()),
                "tail10_eq": borel_tail(10, m),
                "tail10_sim": float((z > 10).mean()),
                "tail50_eq": borel_tail(50, m),
                "tail50_sim": float((z > 50).mean()),
            }
        )
    extinction = []
    for m in params["supercritical_m"]:
        z = simulate_total_progeny(rng, m, params["supercritical_trees"], cap=params["supercritical_cap"])
        extinction.append(
            {
                "m": m,
                "xi_eq": poisson_extinction_probability(m),
                "xi_sim": float((z < params["supercritical_cap"]).mean()),
            }
        )
    overdispersion = []
    m0 = params["dispersion_m"]
    for r in params["dispersions"]:
        z = simulate_total_progeny(rng, m0, params["n_trees"], dispersion=r)
        var_off = m0 + m0 * m0 / r
        overdispersion.append(
            {
                "r": r,
                "offspring_var": var_off,
                "sd_eq": variance_total_progeny(m0, var_off) ** 0.5,
                "sd_sim": float(z.std()),
                "EZ_sim": float(z.mean()),
                "p99_sim": float(np.percentile(z, 99)),
            }
        )
    estimation = []
    for m in params["estimation_m"]:
        for n in params["estimation_n"]:
            m_hat = rng.poisson(m, (params["estimation_reps"], n)).mean(1)
            ok = m_hat < 1
            ez = 1.0 / (1.0 - m_hat[ok])
            estimation.append(
                {
                    "m": m,
                    "n": n,
                    "fraction_subcritical": float(ok.mean()),
                    "mean_abs_rel_error": float(np.mean(np.abs(ez - 1 / (1 - m)) * (1 - m))),
                    "relative_se_eq5": criticality_relative_se(m, n),
                    "ci95_halfwidth_eq5": 1.96 * criticality_relative_se(m, n),
                }
            )
    return {"progeny": progeny, "extinction": extinction, "overdispersion": overdispersion, "estimation": estimation}


# --------------------------------------------------------------------------- Table 4, Fig. 5
def retries_cost(profiles: Profiles, params: dict[str, Any]) -> dict[str, Any]:
    """Success probability and calls per success vs Eqs. (7)-(8) by Monte Carlo over step chains.

    Each step's attempts are drawn once (attempts = min(G, R+1), failure iff G > R+1, with
    G ~ Geometric(1-p)). The paper's script drew attempts and failure from two independent
    geometric draws; both give the same expectations (docs/assumptions.md, A12).
    """
    rng = np.random.default_rng(params.get("seed", 2))
    q = 1.0 / params["mean_steps"]
    rows = []
    for r_max in params["retries"]:
        for p in params["failure_probs"]:
            base = profiles["P3"]
            wf = base.workflow.model_copy(
                update={"mean_steps": params["mean_steps"], "failure_prob": p, "max_retries": r_max}
            )
            sq = structural_quantities(wf)
            n_steps = rng.geometric(q, params["n_tasks"])
            ok = np.ones(n_steps.size, dtype=bool)
            attempts = np.zeros(n_steps.size)
            for j in range(1, int(n_steps.max()) + 1):
                active = ok & (n_steps >= j)
                if not active.any():
                    break
                g = rng.geometric(1.0 - p, int(active.sum())) if p > 0 else np.ones(int(active.sum()), dtype=int)
                attempts[active] += np.minimum(g, r_max + 1)
                ok[np.where(active)[0][g > r_max + 1]] = False
            rows.append(
                {
                    "R": r_max,
                    "p": p,
                    "Ps_eq": sq.success_prob,
                    "Ps_sim": float(ok.mean()),
                    "calls_per_success_eq": sq.calls_per_success,
                    "calls_per_success_sim": float(attempts.mean() / ok.mean()),
                }
            )
    return {
        "rows": rows,
        "max_abs_Ps_error": max(abs(r["Ps_eq"] - r["Ps_sim"]) for r in rows),
        "max_rel_cost_error": max(abs(r["calls_per_success_eq"] / r["calls_per_success_sim"] - 1) for r in rows),
    }


# --------------------------------------------------------------------------- Fig. 6
def fork_join(profiles: Profiles, params: dict[str, Any]) -> dict[str, Any]:
    """Latency vs fixed fan-out K at a fixed arrival rate, with and without the join."""
    opts, no_join = _options(params), _options(params, use_fork_join=False)
    rows = []
    for k in params["fanouts"]:
        system = profiles["P3"].with_updates(
            **{"workflow.fanout_kind": "fixed", "workflow.fanout": k, "tool.servers": params["tool_servers"]}
        )
        model = predict(system, params["rate"], opts)
        nj = predict(system, params["rate"], no_join)
        sim = _sim_chain(system, params["rate"], params, "sd")
        rows.append(
            {
                "K": k,
                "rho_llm": model.rho_llm,
                "rho_tool": model.rho_tool,
                "model": model.to_dict(),
                "no_join": nj.to_dict(),
                "sim": sim,
                "error": _errors(model.to_dict(), sim),
                "no_join_error": _errors(nj.to_dict(), sim),
            }
        )
    return {"rows": rows}


# --------------------------------------------------------------------------- Table 5, Fig. 7
def load_sweep(profiles: Profiles, params: dict[str, Any]) -> dict[str, Any]:
    """Model vs simulation across LLM utilization for each profile."""
    opts = _options(params)
    out: dict[str, Any] = {}
    for name in params["profiles"]:
        system = profiles[name]
        rows = []
        for rho in params["rho_llm"]:
            rate = rate_for_llm_utilization(system, rho)
            model = _pred_dict(system, rate, opts)
            sim = _sim_chain(system, rate, params, "sd")
            rows.append(
                {
                    "rho_llm": rho,
                    "rate": rate,
                    "rho_tool": model["rho_tool"],
                    "model": model,
                    "sim": sim,
                    "error": _errors(model, sim),
                    "success_prob_abs_error": model["success_prob"] - sim["success_prob"][0],
                }
            )
        out[name] = rows
    return out


# --------------------------------------------------------------------------- Table 6
def bottleneck(profiles: Profiles, params: dict[str, Any]) -> dict[str, Any]:
    """Saturation analysis at a fraction of the stability limit (Eq. (10))."""
    opts = _options(params)
    rows = []
    for label, spec in params["cases"].items():
        system = profiles[spec["profile"]]
        if "tool_servers" in spec:
            system = system.with_updates(**{"tool.servers": spec["tool_servers"]})
        lim = stability_limit(system)
        rate = params["fraction_of_limit"] * lim.max_rate
        sq = structural_quantities(system.workflow)
        model = _pred_dict(system, rate, opts)
        sim = _sim_chain(system, rate, params, "sd")
        rows.append(
            {
                "case": label,
                "llm_calls": sq.llm_calls_per_task,
                "tool_calls": sq.tool_calls_per_task,
                "llm_limit": lim.llm_limit,
                "tool_limit": lim.tool_limit,
                "bottleneck": lim.bottleneck,
                "rate": rate,
                "model": model,
                "sim": sim,
                "error": _errors(model, sim),
            }
        )
    return {"rows": rows}


# --------------------------------------------------------------------------- Table 11
def ablation(profiles: Profiles, params: dict[str, Any]) -> dict[str, Any]:
    """Remove one model component at a time (paper Table 11) and compare with simulation."""
    system = profiles["P3"]
    rate = rate_for_llm_utilization(system, params["rho_llm"])
    sim = _sim_chain(system, rate, params, "sd")
    variants: dict[str, tuple[SystemConfig, ModelOptions]] = {
        "full_model": (system, _options(params)),
        "no_queueing": (system, _options(params, use_queueing=False)),
        "no_fork_join": (system, _options(params, use_fork_join=False)),
        "no_retries": (
            system.with_updates(**{"workflow.failure_prob": 0.0, "workflow.max_retries": 0}),
            _options(params),
        ),
        # What the paper's table labels "Exponential service (M/M/k)" was computed as lognormal, CV = 1.
        "lognormal_cv1_paper_exponential_ablation": (
            system.with_updates(**{"llm.cv": 1.0, "tool.cv": 1.0}),
            _options(params),
        ),
        "true_exponential_service": (system, _options(params, service_dist="exponential")),
    }
    rows = {}
    for name, (sys_v, opts) in variants.items():
        pred = predict(sys_v, rate, opts).to_dict()
        rows[name] = {"model": pred, "error": _errors(pred, sim)}
    mv = mean_value_baseline(system, rate)
    rows["mean_value"] = {"model": {"mean": mv}, "error": {"mean": relative_error(mv, sim["mean"][0])}}
    return {"rate": rate, "sim": sim, "variants": rows}


# --------------------------------------------------------------------------- Table 7, Fig. 8
def trees(profiles: Profiles, params: dict[str, Any]) -> dict[str, Any]:
    """Sub-agent trees: model fixed point vs tree simulator."""
    system = profiles["TREE"]
    opts = _options(params)
    ps = structural_quantities(system.workflow).success_prob
    rows = []
    for m in params["delegation_m"]:
        mult = agents_per_tree(m, ps)
        for rho in params["rho_llm"]:
            rate = rate_for_llm_utilization(system, rho, multiplier=mult)
            model = predict_tree(system, rate, m, opts)
            runs = [simulate_tree(system, rate, m, params["n_trees"], seed=s) for s in range(params["seeds"])]
            sim = aggregate(runs, [*LAT, "tree_success", "agents_per_tree"], "ci")
            md = model.to_dict()
            rows.append({"m": m, "rho_llm": rho, "rate": rate, "model": md, "sim": sim, "error": _errors(md, sim)})
    return {"rows": rows}


# --------------------------------------------------------------------------- Table 8
def cv_sensitivity(profiles: Profiles, params: dict[str, Any]) -> dict[str, Any]:
    """Model error as the LLM service-time CV varies (tool CV fixed)."""
    opts = _options(params)
    rows = []
    for cv in params["llm_cv"]:
        system = profiles["P3"].with_updates(**{"llm.cv": cv})
        rate = rate_for_llm_utilization(system, params["rho_llm"])
        model = _pred_dict(system, rate, opts)
        sim = _sim_chain(system, rate, params, "ci")
        rows.append({"llm_cv": cv, "rate": rate, "model": model, "sim": sim, "error": _errors(model, sim)})
    return {"rows": rows}


# --------------------------------------------------------------------------- Table 12
def burst_failure(profiles: Profiles, params: dict[str, Any]) -> dict[str, Any]:
    """Stationary (Poisson) model under MMPP arrivals, with and without the c_a^2 = SCV correction."""
    system = profiles["P3"]
    rows = []
    for rho in params["rho_llm"]:
        rate = rate_for_llm_utilization(system, rho)
        for beta in params["burst_factors"]:
            arrivals = MMPP2(rate, beta, high_exit_rate=params["high_exit_rate"]) if beta > 1 else None
            if arrivals is not None:
                trace = mmpp_arrivals(
                    np.random.default_rng(params.get("scv_seed", 5)), arrivals, params["n_tasks"] / rate
                )
                scv = interarrival_scv(trace)
            else:
                scv = 1.0
            sim = _sim_chain(system, rate, params, "sd", arrivals)
            pois = _pred_dict(system, rate, _options(params))
            corr = _pred_dict(system, rate, _options(params, ca2=scv))
            rows.append(
                {
                    "rho_llm": rho,
                    "burst_factor": beta,
                    "scv": scv,
                    "sim": sim,
                    "poisson_model": pois,
                    "scv_corrected_model": corr,
                }
            )
    return {"rows": rows}


# --------------------------------------------------------------------------- Table 9
def burst_bracket(profiles: Profiles, params: dict[str, Any]) -> dict[str, Any]:
    """Poisson <= simulation <= quasi-static bracket under MMPP arrivals (Eq. (26))."""
    system = profiles["P3"]
    opts = _options(params)
    rows = []
    for sigma in params["high_exit_rates"]:
        for rho in params["rho_llm"]:
            rate = rate_for_llm_utilization(system, rho)
            for beta in params["burst_factors"]:
                arrivals = MMPP2(rate, beta, high_exit_rate=sigma)
                qs, rho_hi = quasi_static_prediction(system, arrivals, opts)
                pois = _pred_dict(system, rate, opts)
                sim = _sim_chain(system, rate, params, "ci", arrivals)
                inside = {
                    k: (pois[k] <= sim[k][0] + sim[k][1]) and (qs is None or sim[k][0] - sim[k][1] <= qs[k])
                    for k in ("mean", "p99")
                }
                rows.append(
                    {
                        "high_exit_rate": sigma,
                        "rho_llm": rho,
                        "burst_factor": beta,
                        "poisson": pois,
                        "quasi_static": qs,
                        "rho_llm_high_state": rho_hi,
                        "sim": sim,
                        "bracket_holds": inside,
                    }
                )
    held = sum(all(r["bracket_holds"].values()) for r in rows)
    return {"rows": rows, "bracket_held": held, "settings": len(rows)}


# --------------------------------------------------------------------------- Table 10
def capacity(profiles: Profiles, params: dict[str, Any]) -> dict[str, Any]:
    """Capacity for a p99 target: rho rule vs model (delta=1) vs derated model, checked by simulation."""
    opts = _options(params)
    rows = []
    for k in params["llm_servers"]:
        system = profiles["P3"].with_updates(**{"llm.servers": k, "tool.servers": params["tool_ratio"] * k})
        naive = rate_for_llm_utilization(system, params["naive_rho"])
        trusted = max_rate_for_p99(system, params["target_p99"], 1.0, opts).max_rate
        derated = max_rate_for_p99(system, params["target_p99"], params["derating"], opts).max_rate
        row: dict[str, Any] = {
            "llm_servers": k,
            "rate_naive": naive,
            "rate_model": trusted,
            "rate_margin": derated,
            "gain_vs_naive": derated / naive - 1.0,
        }
        for label, rate in (("naive", naive), ("model", trusted), ("margin", derated)):
            runs = [simulate_chain(system, rate, params["n_tasks"], seed=s) for s in range(params["seeds"])]
            row[f"sim_p99_{label}"] = mean_ci([r.p99 for r in runs])
        rows.append(row)
    return {"target_p99": params["target_p99"], "rows": rows}


# --------------------------------------------------------------------------- Section 7 text
def depth_sweep(profiles: Profiles, params: dict[str, Any]) -> dict[str, Any]:
    """Model-only sweep of mean workflow length at a fixed arrival rate."""
    opts = _options(params)
    rows = []
    for n_bar in params["mean_steps"]:
        system = profiles["P3"].with_updates(**{"workflow.mean_steps": n_bar})
        pred = predict(system, params["rate"], opts)
        rows.append({"mean_steps": n_bar, **pred.to_dict()})
    return {"rows": rows}


def runtime(profiles: Profiles, params: dict[str, Any]) -> dict[str, Any]:
    """Wall-clock cost of one model evaluation vs one simulation run on THIS machine."""
    system = profiles["P3"]
    rate = rate_for_llm_utilization(system, 0.8)
    opts = _options(params)
    t0 = time.perf_counter()
    for _ in range(params["model_reps"]):
        predict(system, rate, opts)
    model_s = (time.perf_counter() - t0) / params["model_reps"]
    t0 = time.perf_counter()
    simulate_chain(system, rate, params["n_tasks"], seed=1)
    sim_s = time.perf_counter() - t0
    return {
        "model_seconds": model_s,
        "simulation_seconds": sim_s,
        "n_tasks": params["n_tasks"],
        "note": "measured locally; hardware-dependent",
    }


EXPERIMENTS: dict[str, Callable[[Profiles, dict[str, Any]], dict[str, Any]]] = {
    "branching": branching,
    "retries_cost": retries_cost,
    "fork_join": fork_join,
    "load_sweep": load_sweep,
    "bottleneck": bottleneck,
    "ablation": ablation,
    "trees": trees,
    "cv_sensitivity": cv_sensitivity,
    "burst_failure": burst_failure,
    "burst_bracket": burst_bracket,
    "capacity": capacity,
    "depth_sweep": depth_sweep,
    "runtime": runtime,
}
