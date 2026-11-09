# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] - 2026-10-24

Changes for the major revision (reviews in `paper/peer-review.md`). Defaults still reproduce the original manuscript bit for bit.

### Added
- Real-data experiments `real_calibration` and `real_validation` (`docs/real_data.md`, `results/real/full/`). They use AgentInstruct trajectories and the Azure LLM inference trace 2023.
- Azure LLM trace 2024 (HPCA 2025): `agentq.datasets.azure2024` (pyarrow parse + `.npy` cache), experiment `real_azure2024` (week-long burstiness; per-hour model error vs `rho_L x peak ratio`).
- MAST (NeurIPS 2025): `agentq.datasets.mast` (ChatDev/MetaGPT/OpenManus timestamp parsers), experiment `real_mast` (zero-load measured latency vs model, step-time dependence, failure modes). TaskBench evaluated and skipped (synthetic structures).
- BurstGPT (KDD 2025): `agentq.datasets.burstgpt`, experiment `real_burstgpt` (surges over months, failure episodes, measured latency fit, sessions); `pandas` added to the `real` extra.
- SWE-Gym (ICML 2025): `agentq.datasets.swegym` (streamed extraction with cache), experiments `real_swegym` and `real_validation_swegym`; `Trajectory.success` so replays can include failed runs.
- Who&When (ICML 2025): `agentq.datasets.whowhen`, experiment `real_multiagent`; pinned-commit archive download in `fetch_datasets`.
- BFCL (ICML 2025): `agentq.datasets.bfcl` and experiment `real_fanout` (fork-join under the measured fan-out law).
- Mooncake traces (FAST 2025): `agentq.datasets.mooncake`, experiments `real_arrival_traces` and `real_validation_mooncake`; `rate_surges` statistic; per-point progress logging in `real_validation`.
- `agentq.datasets` (loaders, checksummed download via `scripts/fetch_datasets.py`, token-to-latency model) and `agentq.sim.replay` (trace-driven simulator); `real` install extra.
- `ModelOptions.step_tail_tol`: choose `n_max` so the dropped step-count mass is below a tolerance (the fixed 80 underestimates p99 by 7.4% at `n_bar = 16`).
- `ModelOptions.fold_tail` / `convolve(..., fold_tail=True)`: fold mass beyond the grid horizon into the last bin instead of dropping and renormalizing.
- `ModelOptions.tool_ca2` with `"batch"` (`E[K^2]/E[K]`): batch-arrival correction at the tool pool.
- Revision experiments: `load_sweep_ci`, `ablation_ci`, `bottleneck_batch`, `cv_sensitivity_k1`, `cv_runlength`, `depth_validation`.
- `paper/main_rev.tex`, `paper/references_rev.bib`: revised manuscript.

### Fixed
- Tree fixed point docstring now states Eq. (22) as `F_T = F_X * dH` with the atom at zero.

## [1.0.0] - 2026-10-23

### Added
- Analytical model (paper Algorithm 1): structure, queueing, fork-join, retries, step mixture.
- Sub-agent tree fixed point, MMPP quasi-static bracket, SLO capacity search with derating.
- Branching-process closed forms and Monte Carlo.
- Reference simulators for chains (Poisson/MMPP arrivals) and trees, bit-identical to the paper's research code for the same seeds.
- All paper experiments (Tables 3-12, Figs. 4-8) with `full` (paper protocol) and `smoke` modes, and comparison with the paper's raw outputs.
- Validity flags, calibration estimators (synthetic-tested), CLI, Starlette HTTP API, Dockerfile, CI.
