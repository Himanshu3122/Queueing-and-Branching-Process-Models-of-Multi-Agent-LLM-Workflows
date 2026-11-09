"""BurstGPT (KDD 2025): months-long arrivals, load-dependent failures, measured latency, sessions.

Four tests on Azure OpenAI GPT traffic:

1. **Surges over months** - do the rate surges of the one-hour Azure 2023 trace recur? Windows are
   compared both with the global mean and with the mean of their own hour (the fair comparison
   with a one-hour trace replayed at its own mean).
2. **Load-dependent failures** - the model assumes a constant failure probability ``p``; the paper
   names ``p(rho)`` as future work. Failure share vs request rate per 5-minute window.
3. **Measured latency** - ``Elapsed time`` (submission to complete response, whole seconds) is fitted
   with the linear token model used by the replay experiments, and the fit is compared with the
   assumed coefficients. Only conversation-mode requests are used: API-mode elapsed times are zero
   for most short and many long responses (a logging artifact, quantified below).
4. **Sessions** - turns per conversation session vs the geometric law, think time, prompt growth.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from agentq.calibration import fit_service_times
from agentq.datasets.azure import arrival_statistics, rate_surges
from agentq.datasets.burstgpt import load_burstgpt, local_surges, windowed_counts
from agentq.datasets.service import TokenLatencyModel
from agentq.experiments.paper import Profiles
from agentq.experiments.real import _step_diagnostics
from agentq.experiments.real_multiagent import _rate

WIN = 300.0


def _arrivals(df: Any) -> dict[str, Any]:
    t = df["time"].to_numpy()
    hours = ((df["t"].to_numpy() // 3600) % 24).astype(int)
    per_hour = np.bincount(hours, minlength=24) / (t[-1] / 86400.0)
    return {
        "requests": int(t.size),
        "days": float(t[-1] / 86400),
        "arrivals": arrival_statistics(t, windows=(10.0, 60.0, 300.0, 3600.0)),
        "global_surges": rate_surges(t, windows=(60.0, 300.0, 3600.0)),
        "surges_within_hour": local_surges(t, WIN, 3600.0),
        "diurnal_peak_to_trough": float(per_hour.max() / per_hour.min()),
        "hour_of_day_rate": per_hour.tolist(),
    }


def _failures(df: Any) -> dict[str, Any]:
    f = df["failed"].to_numpy()
    out: dict[str, Any] = {
        "overall": _rate(f.tolist()),
        "by_model": {m: float(g.mean()) for m, g in df.groupby("model")["failed"]},
        "by_log_type": {m: float(g.mean()) for m, g in df.groupby("log_type")["failed"]},
        "fail_after_fail": _rate(f[1:][f[:-1]].tolist()),
        "fail_after_success": _rate(f[1:][~f[:-1]].tolist()),
    }
    t = df["time"].to_numpy()
    counts = windowed_counts(t, WIN)
    fails = np.histogram(t[f], bins=np.arange(t[0], t[-1] + WIN, WIN))[0]
    ok = counts >= 20
    rate, share = counts[ok] / WIN, fails[ok] / counts[ok]
    rho, pval = stats.spearmanr(rate, share)
    deciles = np.quantile(rate, np.linspace(0, 1, 11))
    idx = np.clip(np.searchsorted(deciles, rate, side="right") - 1, 0, 9)
    by_dec = [float(fails[ok][idx == i].sum() / counts[ok][idx == i].sum()) for i in range(10)]
    out["load_dependence"] = {
        "window_s": WIN,
        "spearman_rate_vs_failure_share": float(rho),
        "pvalue": float(pval),
        "failure_share_by_rate_decile": by_dec,
        "rate_decile_edges": deciles.tolist(),
        "top_vs_bottom_decile_ratio": by_dec[-1] / by_dec[0] if by_dec[0] > 0 else None,
        "window_failure_share_dispersion": float(share.var() / (share.mean() * (1 - share.mean()) / counts[ok].mean())),
    }
    return out


def _latency(df: Any, assumed: TokenLatencyModel) -> dict[str, Any]:
    ok = df[~df["failed"]]
    api = ok[ok["log_type"] == "API log"]
    long_api = api[api["resp_tokens"] >= 200]
    out: dict[str, Any] = {
        "api_log_artifact": {
            "zero_elapsed_share": float((api["elapsed"] == 0).mean()),
            "zero_elapsed_share_for_200plus_output_tokens": float((long_api["elapsed"] == 0).mean()),
        },
        "assumed_model": assumed.to_dict(),
        "by_model": {},
    }
    conv = ok[ok["log_type"] == "Conversation log"]
    t_all = df["time"].to_numpy()
    counts = windowed_counts(t_all, WIN)
    for model, g in conv.groupby("model"):
        y = g["elapsed"].to_numpy(dtype=float)
        x = np.c_[np.ones(len(g)), g["req_tokens"].to_numpy(), g["resp_tokens"].to_numpy()]
        b = np.linalg.lstsq(x, y, rcond=None)[0]
        pred_fit = x @ b
        pred_assumed = np.array(
            [assumed.service_time(int(r), int(s)) for r, s in zip(g["req_tokens"], g["resp_tokens"], strict=True)]
        )
        long = g["resp_tokens"].to_numpy() >= 50
        per_tok = y[long] / g["resp_tokens"].to_numpy()[long]
        # load dependence: measured / fitted, by request-rate decile of the request's 5-minute window
        w = np.clip(((g["time"].to_numpy() - t_all[0]) // WIN).astype(int), 0, counts.size - 1)
        load = counts[w] / WIN
        ratio = (y + 0.5) / np.maximum(pred_fit, 0.5)  # +0.5: elapsed is truncated to whole seconds
        edges = np.quantile(load, np.linspace(0, 1, 11))
        dec = np.clip(np.searchsorted(edges, load, side="right") - 1, 0, 9)
        rho, pval = stats.spearmanr(load, ratio)
        pos = y[y > 0] + 0.5
        out["by_model"][str(model)] = {
            "requests": int(len(g)),
            "fitted": {
                "overhead_s": float(b[0]),
                "prefill_ms_per_token": float(1000 * b[1]),
                "decode_ms_per_token": float(1000 * b[2]),
                "r2": float(1 - np.var(y - pred_fit) / np.var(y)),
            },
            "assumed_model_r2": float(1 - np.var(y - pred_assumed) / np.var(y)),
            "assumed_model_mean_bias": float(pred_assumed.mean() / y.mean() - 1),
            "measured_ms_per_output_token_median": float(1000 * np.median(per_tok)),
            "elapsed": {"mean": float(y.mean()), "cv": float(y.std() / y.mean()), "p99": float(np.percentile(y, 99))},
            "elapsed_family_aic": [f.to_dict() for f in fit_service_times(pos)] if pos.size >= 10 else [],
            "load_dependence": {
                "spearman_window_rate_vs_measured_over_fitted": float(rho),
                "pvalue": float(pval),
                "median_ratio_by_rate_decile": [float(np.median(ratio[dec == i])) for i in range(10)],
            },
        }
    return out


def _sessions(df: Any) -> dict[str, Any]:
    conv = df[df["session"].notna()].sort_values(["session", "time"], kind="stable")
    sizes = conv.groupby("session").size().to_numpy(dtype=float)
    gaps = conv.groupby("session")["time"].diff().dropna().to_numpy()
    turn = conv.groupby("session").cumcount().to_numpy()
    multi = conv[conv.groupby("session")["session"].transform("size") >= 3]
    rho, _ = stats.spearmanr(multi.groupby("session").cumcount(), multi["req_tokens"])
    first = multi.groupby("session")["req_tokens"].first().to_numpy(dtype=float)
    last = multi.groupby("session")["req_tokens"].last().to_numpy(dtype=float)
    return {
        "sessions": int(sizes.size),
        "turns": _step_diagnostics(sizes, float(sizes.mean())),
        "single_turn_share": float((sizes == 1).mean()),
        "think_time_s": {
            "p10": float(np.percentile(gaps, 10)),
            "p50": float(np.median(gaps)),
            "p90": float(np.percentile(gaps, 90)),
        },
        "max_turn_index": int(turn.max()),
        "prompt_growth": {
            "spearman_turn_vs_request_tokens": float(rho),
            "mean_first_turn_tokens": float(first.mean()),
            "mean_last_turn_tokens": float(last.mean()),
        },
    }


def real_burstgpt(profiles: Profiles, params: dict[str, Any]) -> dict[str, Any]:
    assumed = TokenLatencyModel(**params.get("latency", {}))
    out: dict[str, Any] = {"parts": {}}
    for name, path in params["files"].items():
        df = load_burstgpt(path)
        part: dict[str, Any] = {"arrivals": _arrivals(df), "failures": _failures(df)}
        if "elapsed" in df.columns:
            part["latency"] = _latency(df, assumed)
        if "session" in df.columns:
            part["sessions"] = _sessions(df)
        out["parts"][name] = part
    return out


BURSTGPT_EXPERIMENTS = {"real_burstgpt": real_burstgpt}
