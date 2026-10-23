"""Shared test helpers."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:  # allow `python -m unittest` without installing the package
    sys.path.insert(0, str(SRC))

from agentq.config import SystemConfig, load_profiles  # noqa: E402
from agentq.grid import Grid  # noqa: E402
from agentq.model import ModelOptions  # noqa: E402

PROFILES_PATH = ROOT / "configs" / "profiles.yaml"
EXPERIMENTS_PATH = ROOT / "configs" / "experiments.yaml"
PAPER_DIR = ROOT / "results" / "paper_reported"

# Coarse grid for fast unit tests (300 s horizon); the paper grid is used where results are compared.
FAST = ModelOptions(grid=Grid(dt=0.05, n=6000))


@lru_cache(maxsize=1)
def profiles() -> dict[str, SystemConfig]:
    return load_profiles(PROFILES_PATH)
