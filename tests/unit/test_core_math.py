"""Unit tests for grid, queueing, structure and branching math."""

from __future__ import annotations

import math
import unittest

import numpy as np

from agentq.branching import (
    borel_pmf,
    borel_tail,
    criticality_relative_se,
    expected_total_progeny,
    poisson_extinction_probability,
    simulate_total_progeny,
    variance_total_progeny,
)
from agentq.config import WorkflowConfig
from agentq.grid import Grid, convolve, mean, quantile, summarize
from agentq.queueing import erlang_c, lognormal_params, response_pmf, service_pmf, wait_parameters, wait_pmf
from agentq.structure import pool_loads, rate_for_llm_utilization, stability_limit, structural_quantities
from tests.helpers import profiles  # noqa: F401  (sets sys.path)


def _erlang_c_factorial(k: int, a: float) -> float:
    """Paper's factorial form of Eq. (11), used as an oracle."""
    rho = a / k
    s = sum(a**n / math.factorial(n) for n in range(k))
    t = a**k / (math.factorial(k) * (1 - rho))
    return t / (s + t)


class TestGrid(unittest.TestCase):
    def test_convolution_of_atoms_shifts_mass(self) -> None:
        g = Grid(dt=1.0, n=64)
        a, b = np.zeros(64), np.zeros(64)
        a[3], b[5] = 1.0, 1.0
        c = convolve(a, b)
        self.assertAlmostEqual(c[8], 1.0, places=9)
        self.assertAlmostEqual(mean(c, g), 8.0, places=9)

    def test_quantile_and_summary(self) -> None:
        g = Grid(dt=1.0, n=100)
        pmf = np.full(100, 0.01)
        self.assertEqual(quantile(pmf, g, 0.5), 49.0)
        self.assertEqual(set(summarize(pmf, g)), {"mean", "p50", "p95", "p99"})
        with self.assertRaises(ValueError):
            quantile(pmf, g, 1.5)

    def test_invalid_grid(self) -> None:
        with self.assertRaises(ValueError):
            Grid(dt=0.0)
        with self.assertRaises(ValueError):
            Grid(n=4)


class TestQueueing(unittest.TestCase):
    def test_erlang_c_matches_factorial_form(self) -> None:
        for k, a in [(1, 0.5), (8, 5.5), (16, 3.2), (32, 30.0), (128, 100.0)]:
            self.assertAlmostEqual(erlang_c(k, a), _erlang_c_factorial(k, a), places=12)

    def test_erlang_c_limits(self) -> None:
        self.assertAlmostEqual(erlang_c(1, 0.7), 0.7, places=12)  # M/M/1: P(wait) = rho
        self.assertEqual(erlang_c(4, 4.0), 1.0)
        self.assertEqual(erlang_c(4, 0.0), 0.0)
        self.assertLess(erlang_c(1000, 900.0), 1.0)  # no overflow for large k

    def test_mm1_mean_wait(self) -> None:
        # M/M/1 with lambda=0.5, E[S]=1: E[W] = rho/(mu - lambda) = 1.0
        w = wait_parameters(0.5, 1.0, 1.0, 1, ca2=1.0)
        self.assertIsNotNone(w)
        assert w is not None
        self.assertAlmostEqual(w.mean_wait, 1.0, places=12)
        self.assertIsNone(wait_parameters(1.0, 1.0, 1.0, 1))

    def test_lognormal_service_moments(self) -> None:
        mu, sigma = lognormal_params(3.0, 0.9)
        self.assertAlmostEqual(math.exp(mu + sigma**2 / 2), 3.0, places=12)
        g = Grid(dt=0.01, n=20000)
        pmf = service_pmf(3.0, 0.9, g)
        self.assertAlmostEqual(pmf.sum(), 1.0, places=12)
        self.assertAlmostEqual(mean(pmf, g), 3.0, delta=0.01)

    def test_response_mean_equals_wait_plus_service(self) -> None:
        g = Grid(dt=0.01, n=40000)
        w = wait_parameters(2.0, 3.0, 0.9, 8)
        assert w is not None
        r = response_pmf(2.0, 3.0, 0.9, 8, g)
        assert r is not None
        self.assertAlmostEqual(mean(r, g), w.mean_wait + 3.0, delta=0.02)
        self.assertAlmostEqual(mean(wait_pmf(w, g), g), w.mean_wait, delta=0.01)


class TestStructure(unittest.TestCase):
    def test_cost_per_success_independent_of_retries(self) -> None:
        # Eq. (8): C_s = n_bar / (1 - p) for every R
        for r in range(4):
            wf = WorkflowConfig(mean_steps=5, fanout=2, failure_prob=0.2, max_retries=r)
            self.assertAlmostEqual(structural_quantities(wf).calls_per_success, 5 / 0.8, places=12)

    def test_success_probability_values(self) -> None:
        # Paper Fig. 5 text: p = 0.2, R = 0/1/2 gives P_s ~ 0.44 / 0.83 / 0.96 (n_bar = 5)
        wfs = [WorkflowConfig(mean_steps=5, fanout=2, failure_prob=0.2, max_retries=r) for r in range(3)]
        ps = [structural_quantities(wf).success_prob for wf in wfs]
        self.assertEqual([round(x, 2) for x in ps], [0.44, 0.83, 0.96])

    def test_no_failures(self) -> None:
        sq = structural_quantities(WorkflowConfig(mean_steps=4, fanout=1, failure_prob=0.0, max_retries=0))
        self.assertEqual(sq.success_prob, 1.0)
        self.assertAlmostEqual(sq.llm_calls_per_task, 4.0, places=12)

    def test_loads_and_limits(self) -> None:
        p3 = profiles()["P3"]
        rate = rate_for_llm_utilization(p3, 0.8)
        loads = pool_loads(p3, rate)
        self.assertAlmostEqual(loads.rho_llm, 0.8, places=12)
        lim = stability_limit(p3)
        self.assertEqual(lim.bottleneck, "llm")
        self.assertAlmostEqual(lim.llm_limit, 0.506, places=3)  # Table 6
        p2 = profiles()["P2"].with_updates(**{"tool.servers": 8})
        self.assertEqual(stability_limit(p2).bottleneck, "tool")
        self.assertAlmostEqual(stability_limit(p2).tool_limit, 0.315, places=3)  # Table 6


class TestBranching(unittest.TestCase):
    def test_closed_forms(self) -> None:
        self.assertAlmostEqual(expected_total_progeny(0.9), 10.0, places=12)
        self.assertAlmostEqual(variance_total_progeny(0.9, 0.9) ** 0.5, 30.0, places=9)  # Table 3
        with self.assertRaises(ValueError):
            expected_total_progeny(1.0)

    def test_borel_is_a_distribution(self) -> None:
        total = sum(borel_pmf(n, 0.5) for n in range(1, 400))
        self.assertAlmostEqual(total, 1.0, places=9)
        self.assertAlmostEqual(borel_tail(10, 0.9), 0.173, places=3)  # Table 3

    def test_extinction(self) -> None:
        self.assertEqual(poisson_extinction_probability(0.8), 1.0)
        self.assertAlmostEqual(poisson_extinction_probability(2.0), 0.203, places=3)  # Section 7.1
        xi = poisson_extinction_probability(1.3)
        self.assertAlmostEqual(xi, math.exp(1.3 * (xi - 1)), places=9)

    def test_criticality_se(self) -> None:
        self.assertAlmostEqual(criticality_relative_se(0.8, 500), 0.2, places=2)  # Section 7.1 text

    def test_monte_carlo_matches_mean(self) -> None:
        z = simulate_total_progeny(np.random.default_rng(0), 0.5, 4000)
        self.assertAlmostEqual(z.mean(), 2.0, delta=0.1)


if __name__ == "__main__":
    unittest.main()
