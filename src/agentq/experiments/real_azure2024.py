"""Azure LLM inference trace 2024 (HPCA 2025): a week of arrivals, and whether burstiness predicts error.

1. **Characterization** per service (code, conversation): burstiness from 1 s to 1 h, rate surges
   against the global mean and within each hour, diurnal and day-to-day variation, token lengths.
2. **Does the surge index predict model error?** Every full hour gets a surge index: the share of
   its 1-minute windows above 1.25x the hour's mean rate. Hours are chosen across the index's range
   (quantiles), and the AgentInstruct ``mixed`` workload is replayed with each hour's real arrivals,
   thinned uniformly at random to ``n_tasks`` (thinning keeps the hour's rate profile) and rescaled
   to the target ``rho_L``. The model's error is then related to the index across hours.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import stats

from agentq.calibration import fit_service_times
from agentq.datasets.azure import arrival_statistics, rate_surges, rescale_arrivals
from agentq.datasets.azure2024 import hourly_surge_index, load_azure2024
from agentq.datasets.burstgpt import local_surges
from agentq.datasets.service import TokenLatencyModel
from agentq.experiments.common import relative_error
from agentq.experiments.paper import Profiles, _options, _pred_dict
from agentq.experiments.real import _latency, _replay, _replay_tasks, _summary, _workloads, calibrated_system
from agentq.sim import poisson_arrivals
from agentq.structure import rate_for_llm_utilization

log = logging.getLogger(__name__)


def _characterize(t: np.ndarray, ctx: np.ndarray, gen: np.ndarray, latency: TokenLatencyModel) -> dict[str, Any]:
    hours = (t // 3600).astype(int)
    per_hour = np.bincount(hours)[:-1] / 3600.0 if hours.max() > 0 else np.array([0.0])
    hod = np.array([per_hour[h::24].mean() for h in range(24)])
    days = (t // 86400).astype(int)
    per_day = np.bincount(days)[: int(t[-1] // 86400)]
    service = latency.overhead_s + 1e-3 * (latency.prefill_ms_per_token * ctx + latency.decode_ms_per_token * gen)
    sample = np.random.default_rng(0).choice(service, min(200_000, service.size), replace=False)
    idx = hourly_surge_index(t)
    return {
        "requests": int(t.size),
        "days": float(t[-1] / 86400),
        "mean_rate": float(t.size / t[-1]),
        "arrivals": arrival_statistics(t, windows=(1.0, 10.0, 60.0, 300.0, 3600.0)),
        "global_surges": rate_surges(t, windows=(60.0, 300.0, 3600.0)),
        "surges_within_hour_5min": local_surges(t, 300.0, 3600.0),
        "hourly_surge_index_1min": {
            "p10": float(np.percentile(idx["share_above"], 10)),
            "p50": float(np.median(idx["share_above"])),
            "p90": float(np.percentile(idx["share_above"], 90)),
            "hours_above_zero": float((idx["share_above"] > 0).mean()),
        },
        "diurnal_peak_to_trough": float(hod.max() / hod.min()) if hod.min() > 0 else None,
        "day_to_day_max_over_min": float(per_day.max() / per_day.min()) if per_day.size and per_day.min() > 0 else None,
        "context_tokens": _summary(ctx),
        "generated_tokens": _summary(gen),
        "implied_llm_service": {
            "mean": float(service.mean()),
            "cv": float(service.std() / service.mean()),
            "aic_best": fit_service_times(sample)[0].family,
        },
    }


def _hour_arrivals(t: np.ndarray, hour: int, n: int, rng: np.random.Generator) -> np.ndarray:
    sel = t[(t >= hour * 3600.0) & (t < (hour + 1) * 3600.0)]
    if sel.size > n:
        sel = np.sort(rng.choice(sel, n, replace=False))
    return sel - sel[0]


def real_azure2024(profiles: Profiles, params: dict[str, Any]) -> dict[str, Any]:
    latency = _latency(params)
    services = {name: load_azure2024(path, params["cache_dir"]) for name, path in params["files"].items()}
    out: dict[str, Any] = {"services": {}, "surge_vs_error": {}}
    for name, (t, ctx, gen) in services.items():
        out["services"][name] = _characterize(t, ctx, gen, latency)
        log.info("characterized %s (%d requests)", name, t.size)

    workload = params["workload"]
    trajs = _workloads({**params, "envs": [workload]})[workload]
    system, _ = calibrated_system(trajs, latency, params, workload)
    tasks = _replay_tasks(trajs, latency)
    n, seeds = params["n_tasks"], params["seeds"]
    total = len(services) * len(params["quantiles"]) * len(params["rho_llm"])
    done = 0
    for rho in params["rho_llm"]:
        rate = rate_for_llm_utilization(system, rho)
        model = _pred_dict(system, rate, _options(params))
        poisson = _replay(
            tasks, lambda s, rng, rate=rate: poisson_arrivals(rng, rate, n), system, {**params, "seeds": seeds}
        )
        rows = []
        for name, (t, _ctx, _gen) in services.items():
            idx = hourly_surge_index(t)
            full = idx["requests"] >= n  # hours with enough requests to thin to n
            order = np.argsort(idx["share_above"][full])
            picks = sorted({int(order[round(q * (order.size - 1))]) for q in params["quantiles"]})
            for k in picks:
                hour = int(idx["hour"][full][k])
                ref = _replay(
                    tasks,
                    lambda s, rng, hour=hour, t=t, rate=rate: rescale_arrivals(_hour_arrivals(t, hour, n, rng), rate),
                    system,
                    {**params, "seeds": seeds},
                )
                rows.append(
                    {
                        "service": name,
                        "hour": hour,
                        "surge_share_1min": float(idx["share_above"][full][k]),
                        "peak_ratio_1min": float(idx["peak_ratio"][full][k]),
                        "replay": ref,
                        "error": {q: relative_error(model.get(q), ref[q][0]) for q in ("mean", "p95", "p99")},
                    }
                )
                done += 1
                log.info("progress %d/%d: rho=%.2f %s hour=%d done", done, total, rho, name, hour)
        share = np.array([r["surge_share_1min"] for r in rows])
        p99 = np.array([r["error"]["p99"] for r in rows], dtype=float)
        mean = np.array([r["error"]["mean"] for r in rows], dtype=float)
        rho_p99, pv = stats.spearmanr(share, p99)
        out["surge_vs_error"][str(rho)] = {
            "rate": rate,
            "model": model,
            "poisson_replay": poisson,
            "error_vs_poisson_replay": {q: relative_error(model.get(q), poisson[q][0]) for q in ("mean", "p95", "p99")},
            "hours": rows,
            "spearman_surge_share_vs_p99_error": float(rho_p99),
            "pvalue": float(pv),
            "spearman_surge_share_vs_mean_error": float(stats.spearmanr(share, mean)[0]),
        }
    out["workload"] = {"name": workload, "calibrated_system": system.model_dump()}
    return out


AZURE2024_EXPERIMENTS = {"real_azure2024": real_azure2024, "real_azure2024_empirical": real_azure2024}
