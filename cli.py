"""Command-line interface: ``agentq <command> ...`` (or ``python -m agentq``)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from agentq.config import SystemConfig, load_profiles
from agentq.experiments.paper import EXPERIMENTS
from agentq.experiments.runner import MODES, run_all, run_experiment
from agentq.grid import Grid
from agentq.model import ModelOptions, predict
from agentq.planning import max_rate_for_p99
from agentq.sim import simulate_chain
from agentq.structure import rate_for_llm_utilization, stability_limit
from agentq.validity import check_validity

DEFAULT_PROFILES = "configs/profiles.yaml"
DEFAULT_EXPERIMENTS = "configs/experiments.yaml"


def _system(args: argparse.Namespace) -> SystemConfig:
    profiles = load_profiles(args.profiles)
    if args.profile not in profiles:
        raise SystemExit(f"unknown profile '{args.profile}'; available: {', '.join(sorted(profiles))}")
    return profiles[args.profile]


def _rate(args: argparse.Namespace, system: SystemConfig) -> float:
    if (args.rate is None) == (args.rho is None):
        raise SystemExit("give exactly one of --rate or --rho")
    return args.rate if args.rate is not None else rate_for_llm_utilization(system, args.rho)


def _emit(data: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(data, indent=2))
        return
    for key, value in data.items():
        if isinstance(value, float):
            print(f"{key:>22}: {value:.4g}")
        elif isinstance(value, list):
            print(f"{key:>22}: {len(value)} item(s)")
            for item in value:
                print(f"{'':>24}- {item}")
        else:
            print(f"{key:>22}: {value}")


def cmd_predict(args: argparse.Namespace) -> None:
    system = _system(args)
    rate = _rate(args, system)
    opts = ModelOptions(grid=Grid(dt=args.dt, n=args.points))
    pred = predict(system, rate, opts)
    lim = stability_limit(system)
    out: dict[str, Any] = {
        "profile": args.profile,
        "task_rate": rate,
        **pred.to_dict(),
        "stability_limit": lim.max_rate,
        "bottleneck": lim.bottleneck,
    }
    out["validity_flags"] = [f.code for f in check_validity(system, rate)]
    _emit(out, args.json)


def cmd_simulate(args: argparse.Namespace) -> None:
    system = _system(args)
    rate = _rate(args, system)
    res = simulate_chain(system, rate, args.tasks, seed=args.seed)
    _emit({"profile": args.profile, "task_rate": rate, **res.to_dict()}, args.json)


def cmd_capacity(args: argparse.Namespace) -> None:
    system = _system(args)
    res = max_rate_for_p99(system, args.p99_target, args.derating, ModelOptions(grid=Grid(dt=args.dt, n=args.points)))
    _emit({"profile": args.profile, **res.to_dict()}, args.json)


def cmd_run(args: argparse.Namespace) -> None:
    if args.experiment == "all":
        paths = run_all(args.mode, args.config, args.profiles, args.out)
    else:
        paths = [run_experiment(args.experiment, args.mode, args.config, args.profiles, args.out)]
    for p in paths:
        print(p)


def cmd_compare(args: argparse.Namespace) -> None:
    from agentq.experiments.compare import compare

    path = Path(args.out) / "full" / f"{args.experiment}.json"
    report = compare(args.experiment, path, args.paper_dir)
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(
        f"{report['experiment']}: {report['n_values']} values, max relative difference "
        f"{report['max_relative_difference']:.3g}"
    )
    if args.summary:
        return
    for pair in report["pairs"]:
        print(f"  {pair['label']:<40} paper={pair['paper']:<12.6g} reproduced={pair['reproduced']:.6g}")


def cmd_serve(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SystemExit("the API needs the 'api' extra: pip install -e '.[api]'") from exc
    uvicorn.run("agentq.api.app:app", host=args.host, port=args.port, log_level=args.log_level.lower())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentq", description=__doc__)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = parser.add_subparsers(dest="command", required=True)

    def add_system(p: argparse.ArgumentParser, load: bool = True) -> None:
        p.add_argument("--profiles", default=DEFAULT_PROFILES, help="profiles YAML")
        p.add_argument("--profile", default="P3", help="profile name (default: P3)")
        if load:
            p.add_argument("--rate", type=float, help="task arrival rate (tasks/s)")
            p.add_argument("--rho", type=float, help="target LLM utilization in (0, 1)")
        p.add_argument("--json", action="store_true", help="print JSON")

    def add_grid(p: argparse.ArgumentParser) -> None:
        p.add_argument("--dt", type=float, default=0.02, help="grid step in seconds (paper: 0.02)")
        p.add_argument("--points", type=int, default=30_000, help="grid points (paper: 30000)")

    p = sub.add_parser("predict", help="analytical prediction for a profile")
    add_system(p)
    add_grid(p)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("simulate", help="run the reference simulator once")
    add_system(p)
    p.add_argument("--tasks", type=int, default=30_000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("capacity", help="max task rate for a p99 target")
    add_system(p, load=False)
    add_grid(p)
    p.add_argument("--p99-target", type=float, required=True)
    p.add_argument("--derating", type=float, default=0.85)
    p.set_defaults(func=cmd_capacity)

    p = sub.add_parser("run", help="run a paper experiment")
    p.add_argument("experiment", choices=[*EXPERIMENTS, "all"])
    p.add_argument("--mode", choices=MODES, default="smoke")
    p.add_argument("--config", default=DEFAULT_EXPERIMENTS)
    p.add_argument("--profiles", default=DEFAULT_PROFILES)
    p.add_argument("--out", default="results")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("compare", help="compare a full-mode result with the paper's raw outputs")
    from agentq.experiments.compare import COMPARATORS

    p.add_argument("experiment", choices=sorted(COMPARATORS))
    p.add_argument("--out", default="results", help="results root containing full/<experiment>.json")
    p.add_argument("--paper-dir", default="results/paper_reported")
    p.add_argument("--json", action="store_true")
    p.add_argument("--summary", action="store_true", help="print only the summary line")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("serve", help="start the HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for attr in ("profiles", "config"):
        if hasattr(args, attr) and not Path(getattr(args, attr)).exists():
            print(
                f"error: {attr} file not found: {getattr(args, attr)} (run from the repository root)", file=sys.stderr
            )
            return 2
    try:
        args.func(args)
    except BrokenPipeError:  # output piped into e.g. `head`
        sys.stdout = open(os.devnull, "w")  # noqa: SIM115 - silence the interpreter's flush at exit
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
