"""End-to-end smoke workflow: exercises every public component once with tiny settings.

Usage (from the repository root):  python scripts/smoke_test.py
Outputs are written to results/smoke/. Numbers printed here are smoke-test outputs, NOT paper results.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentq.bursts import MMPP2, bracket  # noqa: E402
from agentq.calibration import calibrate, synthetic_trace  # noqa: E402
from agentq.config import load_profiles  # noqa: E402
from agentq.experiments.runner import run_experiment  # noqa: E402
from agentq.grid import Grid  # noqa: E402
from agentq.model import ModelOptions, predict  # noqa: E402
from agentq.planning import max_rate_for_p99  # noqa: E402
from agentq.sim import simulate_chain, simulate_tree  # noqa: E402
from agentq.structure import rate_for_llm_utilization  # noqa: E402
from agentq.trees import predict_tree  # noqa: E402
from agentq.validity import check_validity  # noqa: E402

OPTS = ModelOptions(grid=Grid(dt=0.05, n=6000))


def main() -> int:
    t0 = time.perf_counter()
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    profiles = load_profiles(ROOT / "configs" / "profiles.yaml")
    check("load profiles", set(profiles) >= {"P1", "P2", "P3", "TREE"}, ", ".join(sorted(profiles)))

    p3 = profiles["P3"]
    rate = rate_for_llm_utilization(p3, 0.5)
    pred = predict(p3, rate, OPTS)
    check("model predict", pred.stable and pred.mean is not None, f"mean={pred.mean:.2f}s p99={pred.p99:.1f}s")

    sim = simulate_chain(p3, rate, 2000, seed=0)
    rel = abs(pred.mean / sim.mean - 1)  # type: ignore[operator]
    check("simulate + agree", rel < 0.15, f"sim mean={sim.mean:.2f}s, |model/sim-1|={rel:.3f} (2000 tasks)")

    tree = predict_tree(profiles["TREE"], 0.2, 0.3, OPTS)
    tsim = simulate_tree(profiles["TREE"], 0.2, 0.3, n_trees=1000, seed=0)
    check("trees", bool(tree.converged), f"model mean={tree.mean:.2f}s sim mean={tsim.mean:.2f}s")

    b = bracket(p3, MMPP2(rate_for_llm_utilization(p3, 0.3), 4.0), OPTS)
    check(
        "burst bracket",
        b.quasi_static is not None,
        f"poisson={b.poisson['mean']:.2f}s qs={b.quasi_static['mean']:.2f}s",
    )

    cap = max_rate_for_p99(p3, 150.0, 0.85, OPTS)
    check("capacity", 0 < cap.max_rate < cap.stability_limit, f"max_rate={cap.max_rate:.3f}/s")

    flags = check_validity(p3, rate_for_llm_utilization(p3, 0.95))
    check("validity flags", any(f.code == "near_saturation" for f in flags), ", ".join(f.code for f in flags))

    est = calibrate(synthetic_trace(p3, 1500, seed=0))
    check("calibration (synthetic)", abs(est.failure_prob - 0.08) < 0.02, f"p_hat={est.failure_prob:.3f}")

    path = run_experiment(
        "depth_sweep", "smoke", ROOT / "configs/experiments.yaml", ROOT / "configs/profiles.yaml", ROOT / "results"
    )
    check("experiment runner", path.is_file(), str(path.relative_to(ROOT)))

    failed = [c for c in checks if not c[1]]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed in {time.perf_counter() - t0:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
