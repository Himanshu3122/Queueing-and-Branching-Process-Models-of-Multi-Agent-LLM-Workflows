"""Fork-join with a real fan-out law (BFCL, ICML 2025).

The model offers two fan-out laws, fixed ``K`` and Poisson ``K``. This experiment measures the
fan-out law of real function-calling queries and tests the fork-join part of the model in
isolation: the reference simulation follows every model assumption (geometric steps, independent
failures, lognormal service, Poisson arrivals) except that each attempt's ``K`` is drawn from the
measured pmf. The model is evaluated with each of its fan-out options, all at the same arrival
rate (set from the true mean ``E[K]``):

* ``poisson``          - Poisson with the true mean;
* ``fixed``            - fixed ``K = round(E[K])`` (the calibration rule for low dispersion);
* ``poisson_batch``    - Poisson, with tool-pool ``c_a^2 = E[K^2]/E[K]`` of the *measured* law;
* ``fixed_batch``      - fixed, with the paper's batch correction ``c_a^2 = K``.

Two pool regimes: LLM-bound (the P3 pools) and tool-bound (``k_L = 32`` and ``k_T`` sized so the
tool pool carries 1.25x the LLM pool's utilization).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from agentq.config import SystemConfig
from agentq.datasets.bfcl import DESIGNED_PARALLEL, LIVE, LIVE_PARALLEL, empirical_pmf, fanout_counts
from agentq.experiments.common import aggregate, relative_error
from agentq.experiments.paper import LAT, Profiles, _options, _pred_dict
from agentq.queueing import lognormal_params
from agentq.sim import ReplayTask, poisson_arrivals, simulate_replay
from agentq.structure import stability_limit

log = logging.getLogger(__name__)
LAWS = {"live_all": LIVE, "live_parallel": LIVE_PARALLEL}
SIM_KEYS = [*LAT, "success_prob", "llm_calls_per_task", "rho_llm", "rho_tool"]


def _law_summary(counts: np.ndarray) -> dict[str, Any]:
    m = float(counts.mean())
    var = float(counts.var(ddof=1))
    lam = m
    ll_poisson = float(sum(k * math.log(lam) - lam - math.lgamma(k + 1) for k in counts))
    pmf = empirical_pmf(counts)
    ll_emp = float(sum(math.log(pmf[int(k)]) for k in counts))
    return {
        "n_queries": int(counts.size),
        "pmf": pmf,
        "mean": m,
        "var": var,
        "dispersion_index": var / m,
        "e_k2_over_e_k": float((counts**2).mean() / m),
        "p_fork": float((counts >= 2).mean()),
        "max": int(counts.max()),
        "loglik_poisson": ll_poisson,
        "loglik_empirical": ll_emp,
        "aic_poisson": 2 * 1 - 2 * ll_poisson,
        "aic_empirical": 2 * (len(pmf) - 1) - 2 * ll_emp,
    }


def _tasks(system: SystemConfig, pmf: dict[int, float], n: int, rng: np.random.Generator) -> list[ReplayTask]:
    """Tasks following the model's workflow law, except K drawn from ``pmf``."""
    wf = system.workflow
    mu, sg = lognormal_params(system.llm.mean_service, system.llm.cv)
    ks, ps = np.array(list(pmf)), np.array(list(pmf.values()))
    tasks: list[ReplayTask] = []
    for _ in range(n):
        attempts: list[tuple[float, int]] = []
        ok = True
        for _step in range(int(rng.geometric(wf.q))):
            for a in range(wf.max_retries + 1):
                attempts.append((float(rng.lognormal(mu, sg)), int(rng.choice(ks, p=ps))))
                if rng.random() >= wf.failure_prob:
                    break
                if a == wf.max_retries:
                    ok = False
            if not ok:
                break
        tasks.append((tuple(attempts), ok))
    return tasks


def _variants(base: SystemConfig, mean_k: float) -> dict[str, tuple[SystemConfig, dict[str, Any]]]:
    poisson = base.with_updates(**{"workflow.fanout_kind": "poisson", "workflow.fanout": mean_k})
    fixed = base.with_updates(**{"workflow.fanout_kind": "fixed", "workflow.fanout": max(1, round(mean_k))})
    return {"poisson": (poisson, {}), "fixed": (fixed, {}), "fixed_batch": (fixed, {"tool_ca2": "batch"})}


def real_fanout(profiles: Profiles, params: dict[str, Any]) -> dict[str, Any]:
    """Measured fan-out laws and the fork-join error they cause."""
    data_dir = params["bfcl_dir"]
    laws = {name: fanout_counts(data_dir, cats) for name, cats in LAWS.items()}
    out: dict[str, Any] = {
        "laws": {name: _law_summary(c) for name, c in laws.items()},
        "designed_parallel_reference": _law_summary(fanout_counts(data_dir, DESIGNED_PARALLEL)),
        "regimes": {},
    }
    base = profiles[params["profile"]]
    total = len(laws) * len(params["regimes"]) * len(params["utilization"])
    done = 0
    for law_name in laws:
        summary = out["laws"][law_name]
        pmf, mean_k = summary["pmf"], summary["mean"]
        for regime, spec in params["regimes"].items():
            k_l = spec["llm_servers"]
            if "tool_servers" in spec:
                k_t = spec["tool_servers"]
            else:  # size the tool pool so its utilization is `tool_to_llm` x the LLM pool's
                ratio = (mean_k * base.tool.mean_service) / (base.llm.mean_service / k_l)
                k_t = max(1, round(ratio / spec["tool_to_llm"]))
            system = base.with_updates(**{"llm.servers": k_l, "tool.servers": k_t})
            variants = _variants(system, mean_k)
            truth = variants["poisson"][0]  # same mean load as the measured law
            limit = stability_limit(truth)
            rows = []
            for u in params["utilization"]:
                rate = u * min(limit.llm_limit, limit.tool_limit)
                runs = []
                for seed in range(params["seeds"]):
                    rng = np.random.default_rng(20_000 + seed)
                    arr = poisson_arrivals(rng, rate, params["n_tasks"])
                    tasks = _tasks(system, pmf, arr.size, rng)
                    runs.append(simulate_replay(tasks, arr, k_l, system.tool, seed=seed))
                sim = aggregate(runs, SIM_KEYS, params.get("spread", "ci"))
                models = {}
                for vname, (vsys, extra) in variants.items():
                    models[vname] = _pred_dict(vsys, rate, _options(params, **extra))
                models["poisson_batch"] = _pred_dict(
                    variants["poisson"][0], rate, _options(params, tool_ca2=summary["e_k2_over_e_k"])
                )
                rows.append(
                    {
                        "bottleneck_utilization": u,
                        "rate": rate,
                        "sim": sim,
                        "model": models,
                        "error": {
                            v: {k: relative_error(m.get(k), sim[k][0]) for k in ("mean", "p95", "p99")}
                            for v, m in models.items()
                        },
                    }
                )
                done += 1
                log.info("progress %d/%d: law=%s regime=%s u=%.2f done", done, total, law_name, regime, u)
            out["regimes"][f"{law_name}/{regime}"] = {
                "llm_servers": k_l,
                "tool_servers": k_t,
                "bottleneck": "tool" if limit.tool_limit < limit.llm_limit else "llm",
                "rows": rows,
            }
    return out


FANOUT_EXPERIMENTS = {"real_fanout": real_fanout}
