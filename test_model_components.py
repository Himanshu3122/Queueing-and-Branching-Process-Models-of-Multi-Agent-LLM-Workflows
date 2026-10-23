"""Unit tests for config, the chain model, trees, bursts, planning, validity and calibration."""

from __future__ import annotations

import unittest

import numpy as np
from pydantic import ValidationError

from agentq.bursts import MMPP2, bracket, interarrival_scv
from agentq.calibration import calibrate, estimate_delegation, fit_service_times, synthetic_trace
from agentq.config import SystemConfig, WorkflowConfig, load_profiles
from agentq.model import ModelOptions, mean_value_baseline, predict
from agentq.planning import max_rate_for_p99, min_llm_servers_for_rate
from agentq.structure import rate_for_llm_utilization
from agentq.trees import agents_per_tree, predict_tree, tree_success_probability
from agentq.validity import check_validity
from tests.helpers import FAST, PROFILES_PATH, profiles


class TestConfig(unittest.TestCase):
    def test_profiles_load(self) -> None:
        p = load_profiles(PROFILES_PATH)
        self.assertEqual(set(p), {"P1", "P2", "P3", "TREE"})
        self.assertEqual(p["P2"].workflow.fanout_kind, "fixed")
        self.assertEqual(p["P2"].tool.servers, 32)

    def test_validation_errors(self) -> None:
        with self.assertRaises(ValidationError):
            WorkflowConfig(mean_steps=5, fanout=2, failure_prob=1.0, max_retries=1)
        with self.assertRaises(ValidationError):
            WorkflowConfig(mean_steps=5, fanout_kind="fixed", fanout=2.5, failure_prob=0.1, max_retries=1)
        with self.assertRaises(ValidationError):
            SystemConfig.model_validate({"workflow": {}, "llm": {}, "tool": {}})
        with self.assertRaises(FileNotFoundError):
            load_profiles("does/not/exist.yaml")

    def test_with_updates(self) -> None:
        p3 = profiles()["P3"]
        q = p3.with_updates(**{"llm.cv": 2.5, "workflow.mean_steps": 8})
        self.assertEqual((q.llm.cv, q.workflow.mean_steps), (2.5, 8))
        self.assertEqual(p3.llm.cv, 0.9)
        with self.assertRaises(KeyError):
            p3.with_updates(**{"llm.nope": 1})

    def test_legacy_roundtrip(self) -> None:
        legacy = {
            "nbar": 5,
            "p": 0.08,
            "R": 1,
            "sL": 3.0,
            "cvL": 0.9,
            "kL": 8,
            "sT": 1.0,
            "cvT": 1.5,
            "kT": 16,
            "fan": "poisson",
            "mu_tool": 2.0,
            "K": None,
        }
        self.assertEqual(SystemConfig.from_legacy(legacy, "P3"), profiles()["P3"])


class TestChainModel(unittest.TestCase):
    def test_prediction_is_consistent(self) -> None:
        p3 = profiles()["P3"]
        pred = predict(p3, rate_for_llm_utilization(p3, 0.5), FAST)
        self.assertTrue(pred.stable)
        assert pred.mean is not None and pred.p95 is not None and pred.p99 is not None and pred.p50 is not None
        self.assertLess(pred.p50, pred.mean)
        self.assertLess(pred.mean, pred.p95)
        self.assertLess(pred.p95, pred.p99)
        self.assertAlmostEqual(pred.pmf.sum(), 1.0, places=9)  # type: ignore[union-attr]
        self.assertLess(pred.horizon_tail_mass, 1e-6)

    def test_unstable_returns_no_latency(self) -> None:
        p3 = profiles()["P3"]
        pred = predict(p3, 10.0, FAST)
        self.assertFalse(pred.stable)
        self.assertIsNone(pred.mean)

    def test_latency_increases_with_load_and_fanout(self) -> None:
        p3 = profiles()["P3"]
        means = [predict(p3, rate_for_llm_utilization(p3, r), FAST).mean for r in (0.3, 0.6, 0.85)]
        self.assertEqual(means, sorted(means))
        fixed = [
            predict(
                p3.with_updates(**{"workflow.fanout_kind": "fixed", "workflow.fanout": k, "tool.servers": 128}),
                0.28,
                FAST,
            ).mean
            for k in (1, 8, 32)
        ]
        self.assertEqual(fixed, sorted(fixed))

    def test_ablations_lower_latency(self) -> None:
        # Table 11: removing queueing, fork-join or retries all reduce predicted latency
        p3 = profiles()["P3"]
        rate = rate_for_llm_utilization(p3, 0.8)
        full = predict(p3, rate, FAST).mean
        g = FAST.grid
        for opts in (ModelOptions(grid=g, use_queueing=False), ModelOptions(grid=g, use_fork_join=False)):
            self.assertLess(predict(p3, rate, opts).mean, full)
        no_retry = p3.with_updates(**{"workflow.failure_prob": 0.0, "workflow.max_retries": 0})
        self.assertLess(predict(no_retry, rate, FAST).mean, full)
        mv = mean_value_baseline(p3, rate)
        self.assertIsNotNone(mv)
        self.assertLess(mv, full)  # type: ignore[operator]

    def test_exponential_service_option(self) -> None:
        p3 = profiles()["P3"]
        rate = rate_for_llm_utilization(p3, 0.5)
        exp_pred = predict(p3, rate, ModelOptions(grid=FAST.grid, service_dist="exponential"))
        self.assertTrue(exp_pred.stable)
        self.assertNotAlmostEqual(exp_pred.mean, predict(p3, rate, FAST).mean, places=3)  # type: ignore[arg-type]


class TestTreesBurstsPlanning(unittest.TestCase):
    def test_tree_identities(self) -> None:
        self.assertAlmostEqual(agents_per_tree(0.5, 0.981), 1 / (1 - 0.5 * 0.981), places=12)
        s = tree_success_probability(0.9, 0.5)
        self.assertAlmostEqual(s, 0.9 * np.exp(0.5 * (s - 1)), places=12)
        with self.assertRaises(ValueError):
            agents_per_tree(1.2, 0.9)

    def test_tree_reduces_to_chain_at_m0(self) -> None:
        t = profiles()["TREE"]
        tree = predict_tree(t, 0.3, 0.0, FAST)
        chain = predict(t, 0.3, FAST)
        assert tree.mean is not None and chain.mean is not None
        self.assertAlmostEqual(tree.mean, chain.mean, delta=1e-6)
        self.assertTrue(tree.converged)
        deeper = predict_tree(t, 0.2, 0.5, FAST)
        self.assertGreater(deeper.mean, predict(t, 0.2, FAST).mean)  # type: ignore[operator]

    def test_mmpp_rates_preserve_mean(self) -> None:
        a = MMPP2(0.3, 4.0)
        self.assertAlmostEqual((1 - a.high_fraction) * a.low_rate + a.high_fraction * a.high_rate, 0.3, places=12)
        self.assertAlmostEqual(a.low_exit_rate, 0.02 / 3, places=12)

    def test_bracket_orders_and_saturation(self) -> None:
        p3 = profiles()["P3"]
        rate = rate_for_llm_utilization(p3, 0.3)
        b = bracket(p3, MMPP2(rate, 8.0), FAST)
        assert b.quasi_static is not None
        self.assertLessEqual(b.poisson["mean"], b.quasi_static["mean"])
        sat = bracket(p3, MMPP2(rate_for_llm_utilization(p3, 0.4), 8.0), FAST)
        self.assertTrue(sat.saturates_in_burst)  # Table 9: rho_L(r1) = 1.16
        self.assertAlmostEqual(sat.rho_llm_high_state, 1.16, places=2)

    def test_scv(self) -> None:
        times = np.cumsum(np.random.default_rng(0).exponential(1.0, 20000))
        self.assertAlmostEqual(interarrival_scv(times), 1.0, delta=0.05)

    def test_capacity_respects_target(self) -> None:
        p3 = profiles()["P3"]
        res = max_rate_for_p99(p3, 150.0, 0.85, FAST)
        assert res.predicted_p99 is not None
        self.assertLessEqual(res.predicted_p99, 0.85 * 150.0)
        self.assertLess(res.max_rate, res.stability_limit)
        trusted = max_rate_for_p99(p3, 150.0, 1.0, FAST)
        self.assertGreater(trusted.max_rate, res.max_rate)
        k = min_llm_servers_for_rate(p3, 0.4, 150.0, 0.85, max_servers=40, tool_ratio=2, options=FAST)
        self.assertIsNotNone(k)


class TestValidityAndCalibration(unittest.TestCase):
    def test_flags(self) -> None:
        p3 = profiles()["P3"]
        self.assertEqual(check_validity(p3, rate_for_llm_utilization(p3, 0.5)), [])
        codes = {f.code for f in check_validity(p3, rate_for_llm_utilization(p3, 0.95), 0.8, 3.0)}
        self.assertTrue({"near_saturation", "delegation_outside_tested", "bursty_arrivals"} <= codes)
        tool_bound = profiles()["P2"].with_updates(**{"tool.servers": 8})
        self.assertIn("tool_bound", {f.code for f in check_validity(tool_bound, 0.25)})
        self.assertEqual(check_validity(p3, 10.0)[0].code, "unstable")

    def test_calibration_recovers_synthetic_parameters(self) -> None:
        p3 = profiles()["P3"]
        rows = synthetic_trace(p3, 4000, seed=1)
        est = calibrate(rows)
        self.assertAlmostEqual(est.failure_prob, 0.08, delta=0.01)
        self.assertAlmostEqual(est.mean_steps, 5.0, delta=0.3)
        self.assertAlmostEqual(est.fanout_mean, 2.0, delta=0.05)
        self.assertEqual(est.fanout_kind, "poisson")
        self.assertEqual(est.llm_fit[0].family, "lognormal")
        self.assertAlmostEqual(est.llm_fit[0].mean, 3.0, delta=0.15)
        sys_est = est.to_system(8, 16)
        self.assertEqual(sys_est.llm.servers, 8)

    def test_fit_and_delegation(self) -> None:
        x = np.random.default_rng(2).exponential(2.0, 3000)
        self.assertIn(fit_service_times(x)[0].family, {"exponential", "gamma"})
        with self.assertRaises(ValueError):
            fit_service_times([1.0, 2.0])
        d = estimate_delegation(np.random.default_rng(3).poisson(0.5, 2000))
        self.assertAlmostEqual(d["m_hat"], 0.5, delta=0.05)
        self.assertAlmostEqual(d["expected_tree_size"], 2.0, delta=0.25)


if __name__ == "__main__":
    unittest.main()
