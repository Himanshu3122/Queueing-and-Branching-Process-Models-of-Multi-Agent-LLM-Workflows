# Implementation status

Paper: `paper/main.tex` (journal version). Engineering specification: `PROJECT_REPORT.md` (source directory). Traceability: `docs/paper_implementation_mapping.md`. Assumptions: `docs/assumptions.md`.

## Fully implemented

- **Analytical model (Algorithm 1, Eqs. (6)-(19)):**
  - retry/success/cost closed forms;
  - pool loads and stability limit;
  - Erlang-C and two-moment waits;
  - response pmfs;
  - fork-join through the fan-out PGF (fixed and Poisson K);
  - retry mixture and success-tilted step mixture;
  - quantiles.
  Outputs match the paper's stored predictions to about 1e-14.
- **Branching results (Eqs. (1)-(5)):** closed forms, extinction, Borel law, criticality standard error, Galton–Watson Monte Carlo (including negative-binomial offspring).
- **Sub-agent trees (Eqs. (20)-(23)):** agents per tree, tree success fixed point, latency fixed point.
- **Bursty arrivals (Eqs. (24)-(26)):** two-state MMPP, quasi-static predictor, Poisson/QS bracket, interarrival SCV.
- **Capacity planning (Eq. (27)):** bisection for the maximum task rate under a derated p99 target.
- **Reference simulators:** chain (Poisson and MMPP arrivals) and tree. Bit-identical to the paper's research code for the same seeds.
- **Experiment protocol:** every results table and figure (Tables 3-12, Figs. 4-8, the §7.9 depth sweep and runtime) as configurable experiments with `full` (paper) and `smoke` modes. JSON results carry parameters and environment metadata.
- **Comparison tooling:** `agentq compare` checks reproduced results value by value against the paper's raw outputs.
- **Interfaces:**
  - CLI: `predict`, `simulate`, `capacity`, `run`, `compare`, `serve`;
  - typed configuration with validation;
  - structured logging.

## Partially implemented

| Component | What is missing / approximated | Reference |
|---|---|---|
| HTTP API | Starlette + Pydantic instead of FastAPI; no authentication | A2, SECURITY.md |
| Ablation "exponential service" | The paper's number was lognormal CV 1; both variants are reported | A6 |
| Retries/cost Monte Carlo | Corrected single-draw version; statistically, not bitwise, equal to the paper's | A12 |
| Calibration | Estimators implemented; tested only on synthetic traces; no trace ingestion pipeline | A1, A13 |
| Validity flags | Thresholds inferred from reported error regimes | A9 |
| Tree simulator | Inherited behaviour: agents of failed trees keep running | A11 |
| Runtime experiment | Hardware-dependent | A15 |
| Docker image | Dockerfile written but **not built** (no Docker daemon in the build environment) | SMOKE_TEST_REPORT.md |

## Not implemented

| Component | Why |
|---|---|
| Calibration and validation on real agent traces | No traces exist in the paper or the source materials |
| Trace-replay arrivals and service in the simulator | Requires trace data and a trace format; listed as future work by the paper and report |
| Load-dependent retries / metastability model | Future work in the paper; no specification to implement |
| Batching-aware LLM service model, Hawkes arrivals, heterogeneous tool pools | Future work in the paper; no specification |
| PostgreSQL, OpenTelemetry ingestion, MLflow/DVC, Ray, web dashboard (suggested in `PROJECT_REPORT.md`) | Not required to implement or reproduce the paper; omitted for dependency discipline (A17) |
| Model training, checkpoints, datasets | The paper's model is analytical; there is nothing to train |

## Technical debt

- **Performance:** each full-grid prediction does about 80 FFT convolutions of length 60,000 (~0.3–1 s), so capacity searches cost about 40 predictions each. A log-spaced grid or caching would speed this up.
- **Fixed horizon:** the grid horizon (600 s) truncates minute-scale tails. `horizon_tail_mass` reports the truncation but does not adapt the grid.
- **RNG stream stability:** simulator reproduction depends on NumPy `Generator` stream stability (verified for NumPy 2.4.4 and 2.5.3). Regression tests will flag changes.
- **Tooling gaps in the build environment:** pytest and FastAPI could not be installed (no PyPI access), so CI (pytest) was not executed locally. The same tests ran with `unittest`.
- **Single-threaded experiments:** parallelism is left to the user (independent experiments can run in parallel shells).
- **Paper build:** the paper copy has not been built with the real IEEEtran class.

## Recommended next steps

1. **Collect real traces** with per-call timing (LLM service and wait, tool spans, retries, spawns). Calibrate with `agentq.calibration` and add a trace-replay mode to the simulator. This is the single most important gap.
2. **Run CI on GitHub** (pytest across Python 3.11–3.13) and build the Docker image.
3. **Fix the dominant model error:** add a batch-arrival correction at the tool pool and a heavier-tailed wait model near saturation (where the paper shows the largest underestimates).
4. **Model load-dependent retries** (timeouts) and detect multiple equilibria (metastability risk).
5. **Run a tool-CV sensitivity sweep.** Only the LLM CV was varied in the paper.

## Research reproducibility

| Claim | Status |
|---|---|
| Model predictions (Tables 5-12, Fig. 6, depth sweep) | **Reproduced locally** to <= 5.3e-14 relative |
| Simulated values (Tables 3, 5-12, Figs. 4, 6-8) | **Reproduced locally**, bit-identical: 299 values across 11 experiments (SMOKE_TEST_REPORT.md) |
| Table 4 Monte Carlo (retries/cost) | Reproduced statistically (corrected Monte Carlo, A12) |
| Runtime numbers | Re-measured locally; not comparable |
| Applicability to real agent systems | **Not established** by the paper or this repository |

## Production-readiness assessment

Scores are out of 5, for a research repository, not a product.

| Dimension | Score | Justification |
|---|---|---|
| Architecture | 4 | Small, layered, pure core; clear mapping to equations; no unnecessary services |
| Code quality | 4 | Typed, documented, ruff-clean, validated configs; a few long numeric functions |
| Maintainability | 4 | Configs separate from code; regression tests pin paper numbers; assumptions documented |
| Scalability | 2 | CPU-bound per-request numerics; no batching, caching or adaptive grid; experiments are sequential |
| Reliability | 3 | Deterministic and well tested in its validated regime; known optimistic tails outside it (flagged, not fixed) |
| Security | 2 | Input validation and bounded request sizes; no authentication, no audit; not security-reviewed |
| Reproducibility | 5 | Seeded, bit-identical to the paper's code where run; provenance in every result file |
| Documentation | 4 | README, eight docs, mapping and assumptions; would benefit from rendered API reference docs |

**Overall:** ready for research use and as a planning aid in its validated regime. Not ready to drive production capacity decisions without trace calibration and validation.
