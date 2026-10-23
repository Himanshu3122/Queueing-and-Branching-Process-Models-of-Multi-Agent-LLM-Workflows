"""Regression against the paper's original research code and its raw outputs.

Golden simulator values were produced by the original scripts (journal_version/simulation/model.py,
model2.py) with the same seeds; the port draws random numbers in the same order, so results must
match exactly. Model predictions are compared with the paper's stored outputs in
results/paper_reported/.
"""

from __future__ import annotations

import json
import unittest

from agentq.bursts import MMPP2
from agentq.model import predict
from agentq.sim import simulate_chain, simulate_tree
from agentq.trees import predict_tree
from tests.helpers import PAPER_DIR, profiles


class TestSimulatorMatchesOriginalCode(unittest.TestCase):
    def test_chain_poisson(self) -> None:
        r = simulate_chain(profiles()["P3"], 0.3, 4000, seed=3)
        self.assertEqual(r.mean, 24.039889650075036)
        self.assertEqual(r.p99, 107.0735167351345)
        self.assertEqual(r.success_prob, 0.9733589673166713)
        self.assertEqual(r.llm_calls_per_task, 5.24525)
        self.assertEqual(r.rho_llm, 0.5989717600404746)

    def test_chain_fixed_fanout(self) -> None:
        self.assertEqual(simulate_chain(profiles()["P2"], 0.3, 3000, seed=1).mean, 24.042806508580245)

    def test_chain_mmpp(self) -> None:
        r = simulate_chain(profiles()["P3"], 0.25, 3000, seed=2, arrivals=MMPP2(0.25, 8, high_exit_rate=0.005))
        self.assertEqual(r.mean, 67.07933713006408)
        self.assertEqual(r.p99, 457.7895510937127)

    def test_tree(self) -> None:
        r = simulate_tree(profiles()["TREE"], 0.25, 0.5, n_trees=3000, seed=4)
        self.assertEqual(r.mean, 25.956766896786046)
        self.assertEqual(r.tree_success, 0.9658840792369773)
        self.assertEqual(r.agents_per_tree, 1.998)


class TestModelMatchesPaperOutputs(unittest.TestCase):
    """Paper grid (dt = 0.02 s, 30,000 points); tolerance covers only floating-point round-off."""

    def test_load_sweep_predictions(self) -> None:
        paper = json.loads((PAPER_DIR / "res1.json").read_text())["e4"]
        for name, idx in (("P3", 3), ("P1", 1)):  # rho_L = 0.8 for P3, 0.5 for P1
            row = paper[name][idx]
            pred = predict(profiles()[name], row["lam"])
            for key in ("mean", "p95", "p99"):
                self.assertAlmostEqual(getattr(pred, key), row["model"][key], delta=1e-9 * row["model"][key])
            self.assertAlmostEqual(pred.success_prob, row["model"]["Psucc"], places=12)

    def test_tree_prediction(self) -> None:
        pred = predict_tree(profiles()["TREE"], 0.25, 0.3)
        self.assertAlmostEqual(pred.mean, 19.273579586852584, delta=1e-9)  # type: ignore[arg-type]
        self.assertEqual(pred.p99, 86.9)
        self.assertEqual(pred.iterations, 11)


if __name__ == "__main__":
    unittest.main()
