# Evaluation

The paper evaluates the analytical model against the discrete-event simulator, not against real systems. The repository reproduces that protocol.

## Metrics (paper Section 6)

| Metric | Definition | Where computed |
|---|---|---|
| Relative error of mean / p95 / p99 | `(model - sim) / sim` for successful-task latency | `experiments.common.relative_error` |
| Absolute error of success probability | `P_s(model) - P_s(sim)` | `load_sweep` rows |
| Error of calls per success | relative error vs Monte Carlo | `retries_cost` |
| Seed spread | sample SD (4-seed experiments) or Student-t 95% half-width (8- and 6-seed experiments) | `mean_sd`, `mean_ci` |

## Experiments

```bash
agentq run <experiment> --mode smoke     # seconds; checks the pipeline only
agentq run <experiment> --mode full      # the paper's protocol
agentq run all --mode full               # everything
agentq compare <experiment>              # full-mode result vs the paper's raw outputs
```

| Experiment | Paper | What it checks |
|---|---|---|
| `branching` | Table 3, Fig. 4 | Total progeny, extinction, overdispersion, criticality-estimation error |
| `retries_cost` | Table 4, Fig. 5 | `P_s` (Eq. 7) and `C_s = n̄/(1-p)` (Eq. 8) |
| `fork_join` | Fig. 6 | Latency vs fixed fan-out K; model without the join |
| `load_sweep` | Table 5, Fig. 7 | Errors vs LLM utilization for P1/P2/P3 |
| `bottleneck` | Table 6 | Errors at 0.9 × λ*; LLM- vs tool-bound |
| `ablation` | Table 11 | Removing queueing, fork-join or retries; distribution variants; mean-value baseline |
| `trees` | Table 7, Fig. 8 | Sub-agent tree fixed point vs tree simulator |
| `cv_sensitivity` | Table 8 | Errors as the LLM service CV varies |
| `burst_failure` | Table 12 | Poisson model (and `c_a^2 = SCV` correction) under MMPP arrivals |
| `burst_bracket` | Table 9 | Poisson ≤ simulation ≤ quasi-static |
| `capacity` | Table 10 | ρ = 0.7 rule vs model (δ = 1) vs derated model (δ = 0.85), checked by simulation |
| `depth_sweep` | §7.9 text | Model-only sweep of n̄ |
| `runtime` | §7.9 text | Model vs simulation wall time on this machine |

## Baselines

All baselines are the paper's; none is an external published system:

- **ρ = 0.7 utilization rule:** a capacity-planning baseline.
- **Ablated models:** without queueing, without fork-join, or without retries.
- **Mean-value calculation:** sums of mean waits and service times.
- **Service-distribution variants:** lognormal with CV 1, as the paper computed its "exponential" row, and true exponential service (added; see A6).

## Output format

Each run writes `results/<mode>/<experiment>.json` containing:

- `params`: the exact merged parameters;
- `environment`: Python, NumPy and SciPy versions and the platform;
- `elapsed_seconds`;
- `provenance`: states that the file was reproduced locally;
- `results`: the data.

Simulation summaries are stored as `[mean, spread]` pairs.

## Reported, reproduced and smoke results

- `results/paper_reported/`: the raw JSON outputs from which the paper's tables were built (original research scripts).
- `results/full/`: files produced by this repository in full mode on the build machine. See `SMOKE_TEST_REPORT.md` for which experiments were run and how they compare.
- `results/smoke/`: tiny runs that only show the pipeline executes. **Not research results.** They are git-ignored.
