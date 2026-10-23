"""Compare locally reproduced experiment results with the paper's original raw outputs.

The paper's tables were generated from the JSON files in ``results/paper_reported/`` (produced by
the original research scripts). This module extracts matching quantities from a reproduced result
file and reports absolute differences, so a researcher can see exactly which numbers reproduce.
Only experiments run in ``full`` mode are comparable; smoke runs use different settings.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

Pair = tuple[str, float, float]  # (label, paper value, reproduced value)


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing result file: {path}")
    return json.loads(path.read_text())


def _load_sweep(paper: dict[str, Any], mine: dict[str, Any]) -> list[Pair]:
    out = []
    for prof, rows in paper["res1"]["e4"].items():
        for pr, mr in zip(rows, mine[prof], strict=True):
            tag = f"{prof} rho={pr['rho']}"
            for k in ("mean", "p95", "p99"):
                out.append((f"{tag} model {k}", pr["model"][k], mr["model"][k]))
                out.append((f"{tag} sim {k}", pr["sim"][k][0], mr["sim"][k][0]))
    return out


def _fork_join(paper: dict[str, Any], mine: dict[str, Any]) -> list[Pair]:
    out = []
    for pr, mr in zip(paper["res1"]["e3"], mine["rows"], strict=True):
        out += [
            (f"K={pr['K']} model mean", pr["m_mean"], mr["model"]["mean"]),
            (f"K={pr['K']} model p99", pr["m_p99"], mr["model"]["p99"]),
            (f"K={pr['K']} no-join mean", pr["n_mean"], mr["no_join"]["mean"]),
            (f"K={pr['K']} sim mean", pr["s_mean"][0], mr["sim"]["mean"][0]),
            (f"K={pr['K']} sim p99", pr["s_p99"][0], mr["sim"]["p99"][0]),
        ]
    return out


def _bottleneck(paper: dict[str, Any], mine: dict[str, Any]) -> list[Pair]:
    out = []
    for pr, mr in zip(paper["res2"]["e7"], mine["rows"], strict=True):
        out += [
            (f"{mr['case']} llm_limit", pr["lamL"], mr["llm_limit"]),
            (f"{mr['case']} model mean", pr["pr"], mr["model"]["mean"]),
            (f"{mr['case']} model p99", pr["pr99"], mr["model"]["p99"]),
            (f"{mr['case']} sim mean", pr["mean_sim"], mr["sim"]["mean"][0]),
            (f"{mr['case']} sim p99", pr["p99_sim"], mr["sim"]["p99"][0]),
        ]
    return out


def _ablation(paper: dict[str, Any], mine: dict[str, Any]) -> list[Pair]:
    names = {
        "Full model": "full_model",
        "No queueing (service only)": "no_queueing",
        "No fork-join (single tool)": "no_fork_join",
        "No retries ($R{=}0,p{=}0$)": "no_retries",
        "Exponential service (M/M/$k$)": "lognormal_cv1_paper_exponential_ablation",
        "Mean-value (no distribution)": "mean_value",
    }
    out = [("sim mean", paper["res2"]["e6_sim"]["mean"][0], mine["sim"]["mean"][0])]
    for row in paper["res2"]["e6"]:
        key = names[row["name"]]
        out.append((f"{key} mean", row["mean"], mine["variants"][key]["model"]["mean"]))
    return out


def _trees(paper: dict[str, Any], mine: dict[str, Any]) -> list[Pair]:
    out = []
    for pr, mr in zip(paper["res_x1"]["x1"], mine["rows"], strict=True):
        tag = f"m={pr['m']} rho={pr['rho']}"
        out += [
            (f"{tag} model mean", pr["model"]["mean"], mr["model"]["mean"]),
            (f"{tag} model p99", pr["model"]["p99"], mr["model"]["p99"]),
            (f"{tag} sim mean", pr["sim"]["mean"][0], mr["sim"]["mean"][0]),
            (f"{tag} sim agents", pr["sim"]["agents"][0], mr["sim"]["agents_per_tree"][0]),
        ]
    return out


def _cv(paper: dict[str, Any], mine: dict[str, Any]) -> list[Pair]:
    out = []
    for pr, mr in zip(paper["res_x2"]["x2"], mine["rows"], strict=True):
        out += [
            (f"cv={pr['cvL']} model mean", pr["model"]["mean"], mr["model"]["mean"]),
            (f"cv={pr['cvL']} sim mean", pr["sim"]["mean"][0], mr["sim"]["mean"][0]),
            (f"cv={pr['cvL']} sim p99", pr["sim"]["p99"][0], mr["sim"]["p99"][0]),
        ]
    return out


def _bracket(paper: dict[str, Any], mine: dict[str, Any]) -> list[Pair]:
    out = []
    for pr, mr in zip(paper["res_x3"]["x3"], mine["rows"], strict=True):
        tag = f"sigma={pr['sw']} rho={pr['rho']} beta={pr['beta']}"
        out.append((f"{tag} sim mean", pr["sim"]["mean"][0], mr["sim"]["mean"][0]))
        out.append((f"{tag} sim p99", pr["sim"]["p99"][0], mr["sim"]["p99"][0]))
        if pr["qs"] is not None and mr["quasi_static"] is not None:
            out.append((f"{tag} QS mean", pr["qs"]["mean"], mr["quasi_static"]["mean"]))
    return out


def _capacity(paper: dict[str, Any], mine: dict[str, Any]) -> list[Pair]:
    out = []
    for pr, mr in zip(paper["res_x4"]["x4"], mine["rows"], strict=True):
        tag = f"kL={pr['kL']}"
        out += [
            (f"{tag} rate model", pr["lam_model"], mr["rate_model"]),
            (f"{tag} rate margin", pr["lam_margin"], mr["rate_margin"]),
            (f"{tag} sim p99 margin", pr["sim_margin"][0], mr["sim_p99_margin"][0]),
        ]
    return out


def _depth(paper: dict[str, Any], mine: dict[str, Any]) -> list[Pair]:
    return [
        (f"nbar={pr['nbar']} mean", pr["mean"], mr["mean"])
        for pr, mr in zip(paper["res2"]["e9"], mine["rows"], strict=True)
    ]


def _branching(paper: dict[str, Any], mine: dict[str, Any]) -> list[Pair]:
    out = []
    for pr, mr in zip(paper["res1"]["e1"], mine["progeny"], strict=True):
        out += [
            (f"m={pr['m']} EZ sim", pr["EZ_sim"], mr["EZ_sim"]),
            (f"m={pr['m']} sd sim", pr["sd_sim"], mr["sd_sim"]),
        ]
    for pr, mr in zip(paper["res1"]["e1b"], mine["extinction"], strict=True):
        out.append((f"m={pr['m']} extinction sim", pr["q_sim"], mr["xi_sim"]))
    return out


def _burst_failure(paper: dict[str, Any], mine: dict[str, Any]) -> list[Pair]:
    out = []
    for pr, mr in zip(paper["res2"]["e5"], mine["rows"], strict=True):
        tag = f"rho={pr['rho']} beta={pr['burst']}"
        out += [
            (f"{tag} sim mean", pr["sim_mean"], mr["sim"]["mean"][0]),
            (f"{tag} sim p99", pr["sim_p99"], mr["sim"]["p99"][0]),
            (f"{tag} poisson mean", pr["pois_mean"], mr["poisson_model"]["mean"]),
            (f"{tag} scv-corrected mean", pr["corr_mean"], mr["scv_corrected_model"]["mean"]),
        ]
    return out


COMPARATORS: dict[str, Callable[[dict[str, Any], dict[str, Any]], list[Pair]]] = {
    "load_sweep": _load_sweep,
    "fork_join": _fork_join,
    "bottleneck": _bottleneck,
    "ablation": _ablation,
    "trees": _trees,
    "cv_sensitivity": _cv,
    "burst_bracket": _bracket,
    "capacity": _capacity,
    "depth_sweep": _depth,
    "burst_failure": _burst_failure,
    "branching": _branching,
}


def compare(name: str, reproduced_file: str | Path, paper_dir: str | Path) -> dict[str, Any]:
    """Compare one reproduced ``full``-mode result with the paper's raw outputs."""
    if name not in COMPARATORS:
        raise KeyError(f"no comparator for '{name}'; available: {sorted(COMPARATORS)}")
    mine = _load(Path(reproduced_file))
    if mine.get("mode") != "full":
        raise ValueError("only full-mode results are comparable with the paper")
    paper_dir = Path(paper_dir)
    stems = ("res1", "res2", "res_x1", "res_x2", "res_x3", "res_x4")
    paper = {stem: _load(paper_dir / f"{stem}.json") for stem in stems}
    pairs = COMPARATORS[name](paper, mine["results"])
    rel = [abs(r / p - 1) for _, p, r in pairs if p not in (0, None) and r is not None]
    return {
        "experiment": name,
        "n_values": len(pairs),
        "max_relative_difference": max(rel) if rel else None,
        "pairs": [{"label": lbl, "paper": p, "reproduced": r} for lbl, p, r in pairs],
    }
