"""General step-count law (``WorkflowConfig.step_pmf``): consistency with the geometric law and basics."""

from __future__ import annotations

import unittest

import numpy as np
from pydantic import ValidationError

import tests.helpers  # noqa: F401  (puts src/ on sys.path)
from agentq.model import ModelOptions, predict
from agentq.sim import simulate_chain, simulate_tree
from agentq.structure import rate_for_llm_utilization, structural_quantities
from tests.helpers import FAST, profiles


def geometric_pmf(mean_steps: float, n_max: int = 4000) -> tuple[float, ...]:
    q = 1.0 / mean_steps
    n = np.arange(1, n_max + 1)
    pmf = q * (1.0 - q) ** (n - 1)
    return tuple(pmf / pmf.sum())


class TestStepLaw(unittest.TestCase):
    def test_explicit_geometric_pmf_reproduces_default_model(self) -> None:
        base = profiles()["P3"]
        explicit = base.with_updates(**{"workflow.step_pmf": geometric_pmf(base.workflow.mean_steps)})
        a, b = structural_quantities(base.workflow), structural_quantities(explicit.workflow)
        for field in ("success_prob", "started_steps", "llm_calls_per_task", "calls_per_success"):
            self.assertAlmostEqual(getattr(a, field), getattr(b, field), places=9)
        rate = rate_for_llm_utilization(base, 0.7)
        opts = ModelOptions(grid=FAST.grid, step_tail_tol=1e-12)
        pa, pb = predict(base, rate, opts), predict(explicit, rate, opts)
        for q in ("mean", "p50", "p95", "p99"):
            self.assertAlmostEqual(getattr(pa, q), getattr(pb, q), delta=1e-6 * getattr(pa, q) + 0.05)

    def test_fixed_length_pipeline(self) -> None:
        base = profiles()["P3"].with_updates(**{"workflow.failure_prob": 0.0})
        fixed = base.with_updates(**{"workflow.step_pmf": (0.0, 0.0, 0.0, 0.0, 1.0)})
        self.assertEqual(fixed.workflow.mean_steps, 5.0)
        sq = structural_quantities(fixed.workflow)
        self.assertAlmostEqual(sq.success_prob, 1.0)
        self.assertAlmostEqual(sq.llm_calls_per_task, 5.0)
        rate = rate_for_llm_utilization(base, 0.3)
        geo, det = predict(base, rate, FAST), predict(fixed, rate, FAST)
        self.assertAlmostEqual(det.mean, geo.mean, delta=0.02 * geo.mean)  # same mean steps
        self.assertLess(det.p99, 0.6 * geo.p99)  # far lighter tail than geometric

    def test_simulator_uses_pmf(self) -> None:
        base = profiles()["P3"].with_updates(**{"workflow.failure_prob": 0.0})
        fixed = base.with_updates(**{"workflow.step_pmf": (0.0, 0.0, 1.0)})
        res = simulate_chain(fixed, 0.05, 2000, seed=1)
        self.assertAlmostEqual(res.llm_calls_per_task, 3.0, places=6)
        with self.assertRaises(NotImplementedError):
            simulate_tree(fixed, 0.01, delegation_mean=0.3, n_trees=50, seed=0)

    def test_validation(self) -> None:
        base = profiles()["P3"]
        with self.assertRaises(ValidationError):
            base.with_updates(**{"workflow.step_pmf": (0.5, 0.4)})


if __name__ == "__main__":
    unittest.main()
