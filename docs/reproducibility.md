# Reproducibility

Two different claims are kept separate:

1. **Software-pipeline reproducibility:** the code installs, runs and produces the paper's experiments deterministically from configuration.
2. **Scientific-result reproduction:** the numbers in the paper's tables come out of this code.

The paper's evidence is itself model-vs-simulation on assumed parameters. Neither claim says anything about real agent systems; no real traces exist (docs/assumptions.md, A1).

## What was reproduced locally (2026-10-23)

The `full`-mode experiments were run on the build machine (2 vCPU, Python 3.13.17, NumPy 2.5.3, SciPy 1.18.1). They were compared with the paper's raw outputs in `results/paper_reported/` using `agentq compare`. Status per experiment is in `SMOKE_TEST_REPORT.md`; the comparison mechanics are:

- **Model predictions:** deterministic; agree with the paper's stored outputs to floating-point round-off (≤ 1e-13 relative).
- **Simulations:** the simulators draw random numbers in the same order as the paper's code, so for the same seeds the outputs are **bit-identical**. They were reproduced under NumPy 2.5.3 although the paper used 2.4.4.
- **`retries_cost`:** uses a corrected single-draw Monte Carlo (A12). Its numbers are statistically, not bitwise, equivalent.
- **`runtime`:** hardware-dependent and not comparable.

## How to reproduce

```bash
pip install -e ".[dev]"
agentq run all --mode full                    # or: make reproduce
make compare                                  # summary line per experiment
agentq compare load_sweep                     # full value-by-value listing
```

Expected wall time on 2 vCPUs is roughly 10–15 minutes in total. The largest experiments are `fork_join`, `capacity`, `burst_bracket`, `load_sweep` and `trees`. Experiments are independent and can be run in parallel shells:

```bash
agentq run load_sweep --mode full & agentq run trees --mode full & wait
```

## Determinism controls

| Control | Where |
|---|---|
| Seeds `0..S-1` per configuration (S = 4, 6 or 8 as in the paper) | `configs/experiments.yaml` (`seeds`) |
| Branching / retries Monte Carlo seeds (1 and 2) | `branching.seed`, `retries_cost.seed` |
| SCV trace seed (5) | `burst_failure.scv_seed` |
| Grid, run length, warm-up | `defaults.full.grid`, `n_tasks`, simulator default `warmup_fraction=0.1` |
| Exact parameters used for a run | `params` field of each result JSON |
| Software environment of a run | `environment` field of each result JSON |

## Known sources of non-reproducibility

- **NumPy random streams:** NumPy does not guarantee `Generator` stream stability across all versions. If a future version changes `geometric`, `poisson` or `lognormal` sampling, simulated numbers will change. `tests/integration/test_reference_regression.py` will then fail loudly, which is intended.
- **Floating-point round-off:** BLAS and FFT backends can perturb model outputs at the 1e-14 level. Quantiles are grid points, so a quantile could in principle shift by one bin (0.02 s) at an exact tie.
- **Timing:** the `runtime` experiment measures the local machine.

## What cannot be reproduced

- Any claim about real agent systems. The model has never been calibrated or validated on traces, and neither the paper nor this repository has trace data.
- The paper's PDF layout and page count, which need the real IEEEtran class (`paper/README.md`).
