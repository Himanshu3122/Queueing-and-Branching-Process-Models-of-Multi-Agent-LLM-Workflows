# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] - 2026-10-24

Changes for the major revision (reviews in `paper/peer-review.md`). Defaults still reproduce the original manuscript bit for bit.

### Added
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
