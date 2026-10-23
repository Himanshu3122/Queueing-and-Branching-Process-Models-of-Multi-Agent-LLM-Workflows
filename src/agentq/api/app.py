"""HTTP API for predictions and capacity planning.

Built on Starlette (the ASGI toolkit FastAPI is built on) with Pydantic validation; FastAPI was
not installable in the build environment (docs/assumptions.md, A2). Endpoints::

    GET  /health            liveness and version
    GET  /profiles          paper profiles available on the server
    POST /predict           chain prediction (+ validity flags)
    POST /predict/tree      sub-agent tree prediction
    POST /capacity          max task rate for a p99 target
    POST /bracket           Poisson / quasi-static bracket for MMPP arrivals
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

import agentq
from agentq.bursts import MMPP2, bracket
from agentq.config import SystemConfig, load_profiles
from agentq.grid import Grid
from agentq.model import ModelOptions, predict
from agentq.planning import max_rate_for_p99
from agentq.structure import rate_for_llm_utilization, structural_quantities
from agentq.trees import agents_per_tree, predict_tree
from agentq.validity import check_validity

log = logging.getLogger(__name__)

MAX_GRID_POINTS = int(os.environ.get("AGENTQ_MAX_GRID_POINTS", "60000"))
DEFAULT_PROFILES = Path(os.environ.get("AGENTQ_PROFILES", "configs/profiles.yaml"))


class GridSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dt: float = Field(0.02, gt=0, le=10)
    n: int = Field(30_000, ge=16)


class SystemRequest(BaseModel):
    """Either a named profile or an inline system, plus a load given as rate or LLM utilization."""

    model_config = ConfigDict(extra="forbid")
    profile: str | None = None
    system: SystemConfig | None = None
    task_rate: float | None = Field(None, gt=0)
    rho_llm: float | None = Field(None, gt=0, lt=1)
    grid: GridSpec = GridSpec()

    @model_validator(mode="after")
    def _one_source(self) -> SystemRequest:
        if (self.profile is None) == (self.system is None):
            raise ValueError("provide exactly one of 'profile' or 'system'")
        if self.grid.n > MAX_GRID_POINTS:
            raise ValueError(f"grid.n must be <= {MAX_GRID_POINTS}")
        return self


class PredictRequest(SystemRequest):
    arrival_scv: float | None = Field(None, gt=0)

    @model_validator(mode="after")
    def _one_load(self) -> PredictRequest:
        if (self.task_rate is None) == (self.rho_llm is None):
            raise ValueError("provide exactly one of 'task_rate' or 'rho_llm'")
        return self


class TreeRequest(PredictRequest):
    delegation_mean: float = Field(ge=0, le=5)


class CapacityRequest(SystemRequest):
    target_p99: float = Field(gt=0)
    derating: float = Field(0.85, gt=0, le=1)


class BracketRequest(PredictRequest):
    burst_factor: float = Field(ge=1, le=100)
    high_fraction: float = Field(0.25, gt=0, lt=1)
    high_exit_rate: float = Field(0.02, gt=0)


def _resolve(req: SystemRequest, profiles: dict[str, SystemConfig]) -> SystemConfig:
    if req.system is not None:
        return req.system
    if req.profile not in profiles:
        raise KeyError(f"unknown profile '{req.profile}'; available: {sorted(profiles)}")
    return profiles[req.profile]


def _rate(req: PredictRequest, system: SystemConfig, multiplier: float = 1.0) -> float:
    if req.task_rate is not None:
        return req.task_rate
    assert req.rho_llm is not None
    return rate_for_llm_utilization(system, req.rho_llm, multiplier)


def _opts(req: SystemRequest) -> ModelOptions:
    return ModelOptions(grid=Grid(dt=req.grid.dt, n=req.grid.n))


def create_app(profiles_path: str | Path = DEFAULT_PROFILES) -> Starlette:
    """Create the ASGI app. Profiles are loaded once; a missing file yields an empty profile set."""
    try:
        profiles = load_profiles(profiles_path)
    except FileNotFoundError:
        log.warning("profiles file %s not found; only inline systems will work", profiles_path)
        profiles = {}

    async def _body(request: Request) -> dict[str, Any]:
        try:
            data = await request.json()
        except ValueError as exc:
            raise ValueError("request body must be valid JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("request body must be a JSON object")
        return data

    def _handle(model: type[BaseModel], fn: Any) -> Any:
        async def endpoint(request: Request) -> JSONResponse:
            try:
                req = model.model_validate(await _body(request))
                return JSONResponse(fn(req))
            except ValidationError as exc:
                detail = exc.errors(include_url=False, include_context=False, include_input=False)
                return JSONResponse({"error": "validation_error", "detail": detail}, status_code=422)
            except (KeyError, ValueError) as exc:
                return JSONResponse({"error": "bad_request", "detail": str(exc)}, status_code=400)
            except Exception:  # pragma: no cover - defensive
                log.exception("unhandled error")
                return JSONResponse({"error": "internal_error"}, status_code=500)

        return endpoint

    def do_predict(req: PredictRequest) -> dict[str, Any]:
        system = _resolve(req, profiles)
        rate = _rate(req, system)
        pred = predict(system, rate, _opts(req))
        flags = check_validity(system, rate, arrival_scv=req.arrival_scv)
        return {"task_rate": rate, "prediction": pred.to_dict(), "validity_flags": [f.to_dict() for f in flags]}

    def do_tree(req: TreeRequest) -> dict[str, Any]:
        system = _resolve(req, profiles)
        mult = agents_per_tree(req.delegation_mean, structural_quantities(system.workflow).success_prob)
        rate = _rate(req, system, mult)
        pred = predict_tree(system, rate, req.delegation_mean, _opts(req))
        # Pool load of a tree workload equals that of chains arriving at rate * M (Eq. (23)).
        flags = check_validity(system, rate * mult, req.delegation_mean, req.arrival_scv)
        return {"task_rate": rate, "prediction": pred.to_dict(), "validity_flags": [f.to_dict() for f in flags]}

    def do_capacity(req: CapacityRequest) -> dict[str, Any]:
        system = _resolve(req, profiles)
        res = max_rate_for_p99(system, req.target_p99, req.derating, _opts(req))
        flags = check_validity(system, res.max_rate)
        return {"capacity": res.to_dict(), "validity_flags": [f.to_dict() for f in flags]}

    def do_bracket(req: BracketRequest) -> dict[str, Any]:
        system = _resolve(req, profiles)
        rate = _rate(req, system)
        arrivals = MMPP2(rate, req.burst_factor, req.high_fraction, req.high_exit_rate)
        return {"task_rate": rate, "bracket": bracket(system, arrivals, _opts(req)).to_dict()}

    async def health(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "version": agentq.__version__})

    async def list_profiles(_: Request) -> JSONResponse:
        return JSONResponse({name: p.model_dump() for name, p in profiles.items()})

    routes = [
        Route("/health", health, methods=["GET"]),
        Route("/profiles", list_profiles, methods=["GET"]),
        Route("/predict", _handle(PredictRequest, do_predict), methods=["POST"]),
        Route("/predict/tree", _handle(TreeRequest, do_tree), methods=["POST"]),
        Route("/capacity", _handle(CapacityRequest, do_capacity), methods=["POST"]),
        Route("/bracket", _handle(BracketRequest, do_bracket), methods=["POST"]),
    ]
    return Starlette(routes=routes)


app = create_app()
