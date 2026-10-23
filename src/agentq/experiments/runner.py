"""Run paper experiments from ``configs/experiments.yaml`` and write results with provenance."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agentq.config import load_profiles, load_yaml
from agentq.experiments.common import Timer, save_result
from agentq.experiments.paper import EXPERIMENTS

log = logging.getLogger(__name__)
MODES = ("full", "smoke")


def experiment_params(config: dict[str, Any], name: str, mode: str) -> dict[str, Any]:
    """Merge ``common`` and mode-specific parameters for one experiment."""
    if name not in EXPERIMENTS:
        raise KeyError(f"unknown experiment '{name}'; choose from {sorted(EXPERIMENTS)}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got '{mode}'")
    spec = config.get("experiments", {}).get(name)
    if spec is None:
        raise KeyError(f"experiment '{name}' missing from experiments config")
    params = dict(config.get("defaults", {}).get(mode, {}))
    params.update(spec.get("common", {}))
    params.update(spec.get(mode, {}))
    return params


def run_experiment(
    name: str, mode: str, config_path: str | Path, profiles_path: str | Path, out_dir: str | Path
) -> Path:
    """Run one experiment and save ``<out_dir>/<mode>/<name>.json``."""
    config = load_yaml(config_path)
    params = experiment_params(config, name, mode)
    profiles = load_profiles(profiles_path)
    log.info("running experiment=%s mode=%s", name, mode)
    with Timer() as timer:
        data = EXPERIMENTS[name](profiles, params)
    path = save_result(Path(out_dir) / mode / f"{name}.json", name, mode, params, data, timer.elapsed)
    log.info("finished experiment=%s in %.1fs -> %s", name, timer.elapsed, path)
    return path


def run_all(
    mode: str, config_path: str | Path, profiles_path: str | Path, out_dir: str | Path, names: list[str] | None = None
) -> list[Path]:
    """Run several (default: all) experiments sequentially."""
    return [run_experiment(n, mode, config_path, profiles_path, out_dir) for n in (names or list(EXPERIMENTS))]
