"""Integration tests: model vs simulator, experiment runner, compare, and the HTTP API."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from agentq.config import load_yaml
from agentq.experiments.compare import compare
from agentq.experiments.runner import experiment_params, run_experiment
from agentq.model import predict
from agentq.sim import simulate_chain
from agentq.structure import rate_for_llm_utilization
from tests.helpers import EXPERIMENTS_PATH, FAST, PAPER_DIR, PROFILES_PATH, profiles


class TestModelAgainstSimulation(unittest.TestCase):
    def test_low_load_agreement(self) -> None:
        """At rho_L = 0.3 the paper reports errors below 0.5%; with short runs allow sampling noise."""
        p3 = profiles()["P3"]
        rate = rate_for_llm_utilization(p3, 0.3)
        pred = predict(p3, rate, FAST)
        runs = [simulate_chain(p3, rate, 4000, seed=s) for s in range(3)]
        sim_mean = sum(r.mean for r in runs) / len(runs)
        sim_ps = sum(r.success_prob for r in runs) / len(runs)
        assert pred.mean is not None
        self.assertLess(abs(pred.mean / sim_mean - 1), 0.08)
        self.assertLess(abs(pred.success_prob - sim_ps), 0.01)
        self.assertAlmostEqual(runs[0].llm_calls_per_task, pred.llm_calls_per_task, delta=0.2)


class TestExperimentRunner(unittest.TestCase):
    def test_param_merge(self) -> None:
        cfg = load_yaml(EXPERIMENTS_PATH)
        full = experiment_params(cfg, "load_sweep", "full")
        self.assertEqual((full["n_tasks"], full["seeds"], full["grid"]["n"]), (30000, 4, 30000))
        smoke = experiment_params(cfg, "bottleneck", "smoke")
        self.assertEqual(list(smoke["cases"]), ["P2_kT8"])
        with self.assertRaises(KeyError):
            experiment_params(cfg, "nope", "full")
        with self.assertRaises(ValueError):
            experiment_params(cfg, "load_sweep", "medium")

    def test_smoke_experiments_write_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("depth_sweep", "retries_cost", "load_sweep"):
                path = run_experiment(name, "smoke", EXPERIMENTS_PATH, PROFILES_PATH, tmp)
                payload = json.loads(path.read_text())
                self.assertEqual(payload["mode"], "smoke")
                self.assertIn("numpy", payload["environment"])
                self.assertTrue(payload["results"])
            rc = json.loads((Path(tmp) / "smoke" / "retries_cost.json").read_text())["results"]
            self.assertLess(rc["max_abs_Ps_error"], 0.05)
            with self.assertRaises(ValueError):  # smoke results must not be compared with the paper
                compare("depth_sweep", Path(tmp) / "smoke" / "depth_sweep.json", PAPER_DIR)


def _asgi_call(app: Any, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    """Minimal in-process ASGI client (avoids an httpx dependency)."""
    raw = json.dumps(body).encode() if body is not None else b""
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("test", 1),
        "server": ("test", 80),
        "scheme": "http",
        "root_path": "",
    }
    messages = [{"type": "http.request", "body": raw, "more_body": False}]
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    payload = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return status, json.loads(payload)


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from agentq.api.app import create_app

        cls.app = create_app(PROFILES_PATH)

    def test_health_and_profiles(self) -> None:
        status, body = _asgi_call(self.app, "GET", "/health")
        self.assertEqual((status, body["status"]), (200, "ok"))
        status, body = _asgi_call(self.app, "GET", "/profiles")
        self.assertIn("P3", body)

    def test_predict(self) -> None:
        req = {"profile": "P3", "rho_llm": 0.92, "grid": {"dt": 0.05, "n": 6000}}
        status, body = _asgi_call(self.app, "POST", "/predict", req)
        self.assertEqual(status, 200)
        self.assertTrue(body["prediction"]["stable"])
        self.assertIn("near_saturation", [f["code"] for f in body["validity_flags"]])

    def test_tree_capacity_bracket(self) -> None:
        g = {"dt": 0.05, "n": 6000}
        status, body = _asgi_call(
            self.app, "POST", "/predict/tree", {"profile": "TREE", "rho_llm": 0.5, "delegation_mean": 0.3, "grid": g}
        )
        self.assertEqual(status, 200)
        self.assertGreater(body["prediction"]["agents_per_tree"], 1.0)
        status, body = _asgi_call(self.app, "POST", "/capacity", {"profile": "P3", "target_p99": 150, "grid": g})
        self.assertEqual(status, 200)
        self.assertGreater(body["capacity"]["max_rate"], 0)
        status, body = _asgi_call(
            self.app, "POST", "/bracket", {"profile": "P3", "rho_llm": 0.3, "burst_factor": 4, "grid": g}
        )
        self.assertEqual(status, 200)
        self.assertFalse(body["bracket"]["saturates_in_burst"])

    def test_validation_errors(self) -> None:
        status, _ = _asgi_call(self.app, "POST", "/predict", {"profile": "P3"})
        self.assertEqual(status, 422)
        status, _ = _asgi_call(self.app, "POST", "/predict", {"profile": "P3", "rho_llm": 0.5, "grid": {"n": 10**7}})
        self.assertEqual(status, 422)
        status, body = _asgi_call(self.app, "POST", "/predict", {"profile": "ZZ", "rho_llm": 0.5})
        self.assertEqual((status, body["error"]), (400, "bad_request"))


if __name__ == "__main__":
    unittest.main()
