"""Smoke tests: the CLI initializes and executes minimal workflows end to end."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from agentq.cli import main
from tests.helpers import EXPERIMENTS_PATH, PROFILES_PATH


def _run(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = main(argv)
    return code, buf.getvalue()


class TestCliSmoke(unittest.TestCase):
    def test_predict_json(self) -> None:
        code, out = _run(
            [
                "--log-level",
                "WARNING",
                "predict",
                "--profiles",
                str(PROFILES_PATH),
                "--profile",
                "P3",
                "--rho",
                "0.5",
                "--dt",
                "0.05",
                "--points",
                "6000",
                "--json",
            ]
        )
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertTrue(data["stable"])
        self.assertEqual(data["bottleneck"], "llm")

    def test_simulate(self) -> None:
        code, out = _run(
            [
                "--log-level",
                "WARNING",
                "simulate",
                "--profiles",
                str(PROFILES_PATH),
                "--rho",
                "0.5",
                "--tasks",
                "500",
                "--json",
            ]
        )
        self.assertEqual(code, 0)
        self.assertGreater(json.loads(out)["mean"], 0)

    def test_capacity(self) -> None:
        code, out = _run(
            [
                "--log-level",
                "WARNING",
                "capacity",
                "--profiles",
                str(PROFILES_PATH),
                "--p99-target",
                "150",
                "--dt",
                "0.05",
                "--points",
                "6000",
                "--json",
            ]
        )
        self.assertEqual(code, 0)
        self.assertGreater(json.loads(out)["max_rate"], 0)

    def test_run_smoke_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, out = _run(
                [
                    "--log-level",
                    "WARNING",
                    "run",
                    "depth_sweep",
                    "--mode",
                    "smoke",
                    "--config",
                    str(EXPERIMENTS_PATH),
                    "--profiles",
                    str(PROFILES_PATH),
                    "--out",
                    tmp,
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue(Path(out.strip()).is_file())

    def test_missing_config_is_reported(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            code, _ = _run(["predict", "--profiles", "missing.yaml", "--rho", "0.5"])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
