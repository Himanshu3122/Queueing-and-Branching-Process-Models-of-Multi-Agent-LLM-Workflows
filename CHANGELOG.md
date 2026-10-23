# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2026-10-23

### Added
- Analytical model (paper Algorithm 1): structure, queueing, fork-join, retries, step mixture.
- Sub-agent tree fixed point, MMPP quasi-static bracket, SLO capacity search with derating.
- Branching-process closed forms and Monte Carlo.
- Reference simulators for chains (Poisson/MMPP arrivals) and trees, bit-identical to the paper's research code for the same seeds.
- All paper experiments (Tables 3-12, Figs. 4-8) with `full` (paper protocol) and `smoke` modes, and comparison with the paper's raw outputs.
- Validity flags, calibration estimators (synthetic-tested), CLI, Starlette HTTP API, Dockerfile, CI.
