"""Typed configuration for workflows, pools and the full system (paper Section 3, Table 2).

Symbol mapping to the paper:

=====================  ==========================================
Field                  Paper symbol
=====================  ==========================================
``mean_steps``         ``n_bar = 1/q`` (geometric steps per task)
``fanout_kind``        fixed ``K`` or Poisson with mean ``mu_K``
``fanout``             ``K`` (fixed) or ``mu_K`` (Poisson)
``failure_prob``       ``p`` (per-attempt failure probability)
``max_retries``        ``R``
``llm.servers``        ``k_L``;  ``llm.mean_service`` = ``E[S_L]``
``tool.servers``       ``k_T``;  ``tool.mean_service`` = ``E[S_T]``
``*.cv``               coefficient of variation of service time
=====================  ==========================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

FanoutKind = Literal["poisson", "fixed"]


class PoolConfig(BaseModel):
    """A pool of identical FIFO servers (LLM endpoints or tool workers)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    servers: int = Field(gt=0, le=100_000, description="Number of parallel servers k")
    mean_service: float = Field(gt=0, description="Mean service time E[S] in seconds")
    cv: float = Field(gt=0, le=20, description="Coefficient of variation of the service time")


class WorkflowConfig(BaseModel):
    """Structure of one agent chain (paper Section 3.1)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mean_steps: float = Field(ge=1.0, le=10_000, description="n_bar: mean of geometric step count")
    fanout_kind: FanoutKind = "poisson"
    fanout: float = Field(ge=0, le=10_000, description="K (fixed) or mu_K (Poisson)")
    failure_prob: float = Field(ge=0.0, lt=1.0, description="p: per-attempt failure probability")
    max_retries: int = Field(ge=0, le=100, description="R: retries allowed per step")

    @model_validator(mode="after")
    def _fixed_fanout_is_integer(self) -> WorkflowConfig:
        if self.fanout_kind == "fixed" and float(self.fanout) != int(self.fanout):
            raise ValueError("fixed fan-out K must be an integer")
        return self

    @property
    def q(self) -> float:
        """Per-step termination probability ``q = 1 / n_bar``."""
        return 1.0 / self.mean_steps


class SystemConfig(BaseModel):
    """Workflow plus the two shared pools."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = ""
    workflow: WorkflowConfig
    llm: PoolConfig
    tool: PoolConfig

    def with_updates(self, **changes: Any) -> SystemConfig:
        """Return a copy with dotted-path updates, e.g. ``with_updates(**{"llm.cv": 2.5})``."""
        data = self.model_dump()
        for path, value in changes.items():
            node = data
            parts = path.split(".")
            for part in parts[:-1]:
                if part not in node or not isinstance(node[part], dict):
                    raise KeyError(f"unknown config path: {path}")
                node = node[part]
            if parts[-1] not in node:
                raise KeyError(f"unknown config path: {path}")
            node[parts[-1]] = value
        return SystemConfig.model_validate(data)

    @classmethod
    def from_legacy(cls, cfg: dict[str, Any], name: str = "") -> SystemConfig:
        """Build from the flat dict used by the original research scripts.

        Keys: ``nbar, p, R, sL, cvL, kL, sT, cvT, kT, fan, mu_tool, K``.
        """
        fan = cfg["fan"]
        fanout = cfg["mu_tool"] if fan == "poisson" else cfg["K"]
        return cls(
            name=name,
            workflow=WorkflowConfig(
                mean_steps=cfg["nbar"],
                fanout_kind=fan,
                fanout=fanout,
                failure_prob=cfg["p"],
                max_retries=cfg["R"],
            ),
            llm=PoolConfig(servers=cfg["kL"], mean_service=cfg["sL"], cv=cfg["cvL"]),
            tool=PoolConfig(servers=cfg["kT"], mean_service=cfg["sT"], cv=cfg["cvT"]),
        )


def load_profiles(path: str | Path) -> dict[str, SystemConfig]:
    """Load named system profiles from a YAML file (see ``configs/profiles.yaml``)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"profile file not found: {path}")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or "profiles" not in raw:
        raise ValueError(f"{path} must contain a top-level 'profiles' mapping")
    return {name: SystemConfig.model_validate({"name": name, **body}) for name, body in raw["profiles"].items()}


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping with a clear error if the file is missing or malformed."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data
