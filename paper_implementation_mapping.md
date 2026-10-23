# Paper-to-repository mapping

Paper: `paper/main.tex` (journal version). Equation, table and figure numbers refer to its compiled form.

**Status legend:**
- ✅ implemented as specified
- 🟡 implemented with assumptions or approximations (see `docs/assumptions.md`)
- ⛔ not implementable from the available materials

## Model (Sections 3–5)

| Paper concept | Equations | Module / function | Status | Notes |
|---|---|---|---|---|
| Workflow: geometric steps, fan-out, retries | §3.1 | `config.WorkflowConfig` | ✅ | Pydantic validation |
| Profiles P1/P2/P3 and tree chain | §3.2, Table 2 | `configs/profiles.yaml` | ✅ | Assumed values, as in the paper |
| Branching: E[Z], Var[Z] | (1)-(2) | `branching.expected_total_progeny`, `variance_total_progeny` | ✅ | |
| Extinction probability | (3) | `branching.poisson_extinction_probability` | ✅ | Poisson offspring |
| Borel law | (4) | `branching.borel_pmf`, `borel_tail` | ✅ | |
| Criticality estimation error | (5) | `branching.criticality_relative_se`, `calibration.estimate_delegation` | ✅ | |
| Retries, success, cost per success | (6)-(8) | `structure.structural_quantities` | ✅ | `C_s = n̄/(1-p)` unit-tested |
| Pool loads, stability limit | (9)-(10) | `structure.pool_loads`, `stability_limit` | ✅ | |
| Erlang-C | (11) | `queueing.erlang_c` | 🟡 | Stable recursion (A4) |
| Two-moment mean wait | (12) | `queueing.wait_parameters` | ✅ | `c_a^2` configurable |
| Wait distribution | (13) | `queueing.wait_pmf` | ✅ | Atom + exponential |
| Response = wait * service | (14) | `queueing.response_pmf` | ✅ | |
| Fork-join via fan-out PGF | (15) | `model._join_pmf` | ✅ | Fixed and Poisson K |
| Attempt, step, task distributions | (16)-(18) | `model.chain_prediction` | ✅ | n_max = 80 |
| Quantiles | (19) | `grid.quantile` | ✅ | |
| Algorithm 1 | Alg. 1 | `model.predict` | ✅ | Matches the paper's outputs to about 1e-14 |
| Agents per tree, tree success | (20)-(21) | `trees.agents_per_tree`, `tree_success_probability` | ✅ | |
| Tree latency fixed point | (22)-(23) | `trees.predict_tree` | ✅ | Same tolerance (1e-7) and iteration cap |
| MMPP rates | (24) | `bursts.MMPP2` | ✅ | |
| Quasi-static predictor | (25) | `bursts.quasi_static_prediction` | ✅ | |
| Poisson / QS bracket | (26) | `bursts.bracket` | ✅ | Empirical regularity, not a theorem |
| SLO capacity with derating | (27) | `planning.max_rate_for_p99` | ✅ | 40-step bisection |

## Simulation and experimental protocol (Section 6, Table 2)

| Paper element | Module | Status | Notes |
|---|---|---|---|
| Heap-based discrete-event simulator, FIFO k-server pools, lognormal service | `sim.chain.simulate_chain` | ✅ | Same RNG draw order as the paper's code; bit-identical outputs for the same seeds |
| MMPP arrivals | `sim.arrivals.mmpp_arrivals` | ✅ | Bit-identical |
| Tree simulator | `sim.tree.simulate_tree` | 🟡 | Bit-identical; inherited failure behaviour (A11) |
| 30,000 tasks, 10% warm-up, 4 seeds (8 or 6 for journal additions) | `configs/experiments.yaml` (full) | ✅ | |
| Grid dt = 0.02 s, 30,000 points, n_max = 80 | `grid.Grid`, `ModelOptions` | ✅ | |
| 95% Student-t CIs | `experiments.common.mean_ci` | ✅ | |
| Hardware (1 vCPU) | — | 🟡 | Timing is machine-dependent (A15) |

## Results (Sections 7–8)

| Table / figure | Experiment (`agentq run <name>`) | Status | Notes |
|---|---|---|---|
| Table 3, Fig. 4 (branching) | `branching` | ✅ | Same seed and draw order |
| Table 4, Fig. 5 (retries, cost) | `retries_cost` | 🟡 | Single-draw Monte Carlo (A12) |
| Fig. 6 (fork-join sweep) | `fork_join` | ✅ | |
| Table 5, Fig. 7 (load sweep) | `load_sweep` | ✅ | |
| Table 6 (bottleneck) | `bottleneck` | ✅ | |
| Table 11 (ablation) | `ablation` | 🟡 | Adds a true exponential variant (A6) |
| Table 7, Fig. 8 (trees) | `trees` | ✅ | |
| Table 8 (CV sensitivity) | `cv_sensitivity` | ✅ | |
| Table 12 (Poisson model under MMPP) | `burst_failure` | ✅ | |
| Table 9 (bracket) | `burst_bracket` | ✅ | |
| Table 10 (capacity) | `capacity` | ✅ | |
| Depth sweep (§7.9 text) | `depth_sweep` | ✅ | |
| Runtime (§7.9 text) | `runtime` | 🟡 | Hardware-dependent |
| Comparison with the paper's raw outputs | `agentq compare <name>` | ✅ | Uses `results/paper_reported/` |

## Discussion, limitations, future work (Sections 9–10)

| Paper element | Module | Status | Notes |
|---|---|---|---|
| Deployment workflow: estimate parameters → λ* → bracket → size with margin | `calibration`, `structure`, `bursts`, `planning`, CLI/API | 🟡 | Estimators tested on synthetic data only |
| MLE + AIC distribution selection | `calibration.fit_service_times` | 🟡 | Lognormal / exponential / gamma |
| Report m with an interval | `calibration.estimate_delegation` | ✅ | |
| Validity regions (where the model fails) | `validity.check_validity` | 🟡 | Thresholds inferred from reported errors (A9) |
| Calibration and validation on real agent traces | — | ⛔ | No traces exist (A1) |
| Load-dependent retries / metastability | — | ⛔ | Future work in the paper; not modeled |
| Batching-aware LLM service, Hawkes arrivals | — | ⛔ | Future work in the paper |
| Real IEEEtran build of the paper | — | ⛔ | Outside this repository's scope |

## Items from `PROJECT_REPORT.md`

| Report item | Status | Notes |
|---|---|---|
| Library layout (structure, queueing, compose, trees, bursts, planning) | ✅ | Composition lives in `model.py` |
| Reference simulator | ✅ | |
| API (`/predict`, `/capacity`, `/bracket`, validity flags) | 🟡 | Starlette instead of FastAPI (A2); adds `/predict/tree` and `/profiles` |
| Validity-region monitor | 🟡 | Configuration-level checks; there is no live telemetry ingestion |
| Trace calibration pipeline | 🟡 | Estimators only; no OpenTelemetry ingestion |
| PostgreSQL, MLflow/DVC, Ray, dashboard | ⛔ by choice | Not needed (A17) |
