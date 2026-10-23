# Smoke test report

Every result below comes from a command actually executed while building the repository on 2026-10-23. Nothing is extrapolated. Failures and blocked checks are recorded alongside the passes.

## Environment

| Item | Value |
|---|---|
| OS | Linux 6.18.44 x86_64 (glibc 2.39) |
| CPU | Intel Xeon @ 2.80 GHz, 2 vCPU, no GPU |
| Python | 3.13.17 |
| numpy / scipy | 2.5.3 / 1.18.1 |
| pydantic / PyYAML | 2.13.5 / 6.0.3 |
| starlette / uvicorn | 1.7.0 / 0.54.0 |
| ruff | 0.16.9 |
| PyPI access | **Unavailable**: proxy refused `pypi.org`. `fastapi`, `pytest` and `httpx` could not be installed |
| Docker | CLI present, **no daemon** |

## Summary

| Status | Count | Items |
|---|---|---|
| PASS | 16 | Imports and install, config validation, lint, format, unit, integration, smoke tests, smoke script, all smoke experiments, CLI, API (in-process and live server), full-mode reproduction of 11 experiments, statistical reproduction of 1, runtime measurement |
| FAIL (fixed) | 5 | See "Failures encountered and fixes" |
| NOT RUN | 2 | pytest runner, GitHub Actions CI |
| BLOCKED | 2 | Docker image build, FastAPI |

## Checks

| # | Test | Status | Command | Observed result | Notes |
|---|---|---|---|---|---|
| 1 | Package install and console entry point | PASS | `python3 -m venv --system-site-packages venv && venv/bin/pip install --no-deps --no-build-isolation -e . && venv/bin/agentq --help` | Installed; `agentq` lists 6 subcommands | `--no-deps` because PyPI was unreachable; dependencies were preinstalled |
| 2 | Import validation | PASS | (part of every test run) | All modules import | |
| 3 | Config validation | PASS | `unittest` `TestConfig` | Profiles load; invalid p, non-integer fixed K, empty system and missing file rejected | |
| 4 | Lint | PASS | `ruff check src tests scripts` | All checks passed | |
| 5 | Formatting | PASS | `ruff format --check src tests scripts` | 36 files already formatted | |
| 6 | Unit tests | PASS | `PYTHONPATH=src python3 -m unittest discover -s tests/unit -t .` | 35 tests OK | |
| 7 | Integration tests | PASS | `... -s tests/integration -t .` | 13 tests OK | Includes bit-exact regression vs the paper's code and the API |
| 8 | Smoke tests | PASS | `... -s tests/smoke -t .` | 5 tests OK | CLI end to end |
| 9 | Full suite (final run) | PASS | `PYTHONPATH=src python3 -m unittest discover -s tests -t .` | **53 tests OK in 9.1 s** | |
| 10 | Smoke workflow script | PASS | `python3 scripts/smoke_test.py` | **9/9 checks passed in 1.9 s** | Outputs are smoke values, not paper results |
| 11 | All experiments, smoke mode | PASS | `python3 -m agentq run all --mode smoke` | 13 result files in 7.3 s | |
| 12 | CLI predict | PASS | `agentq predict --profile P3 --rho 0.8` | mean 27.11 s, p99 120 s, P_s 0.9688, bottleneck llm | Equals the paper's Table 5 model value |
| 13 | API (in-process ASGI) | PASS | `TestApi` (4 tests) | health, profiles, predict, tree, capacity, bracket, 422/400 errors | |
| 14 | API (live server) | PASS | `python3 -m agentq serve --port 8765` + `curl` | `/health` ok; `/predict` ρ=0.9 → p99 156.74 s with flag; `/capacity` → 0.4231/s (paper Table 10: 0.4231); invalid body → 422; non-JSON → 400 | Server stopped by a 60 s `timeout` (exit 124), as intended |
| 15 | Full-mode reproduction vs paper raw outputs | PASS | `agentq run <exp> --mode full`; `agentq compare <exp>` | See next table | |
| 16 | Runtime experiment | PASS | `agentq run runtime --mode full` | 0.216 s per prediction; 1.44 s per 30,000-task simulation | Paper reported 0.52 s / 2.2 s on its machine; not comparable |
| 17 | pytest runner | NOT RUN | `pytest` | — | pytest not installable (no PyPI). Tests are pytest-compatible `unittest` classes |
| 18 | GitHub Actions CI | NOT RUN | `.github/workflows/ci.yml` | — | Requires GitHub |
| 19 | Docker image build | BLOCKED | `docker build -t agentq:test .` | `failed to connect to the docker API at unix:///var/run/docker.sock` | No daemon in the build environment |
| 20 | FastAPI | BLOCKED | `pip install fastapi` | `No matching distribution found` (proxy 403 on pypi.org) | Starlette used instead (docs/assumptions.md, A2) |

## Full-mode reproduction (paper protocol, same seeds)

Command: `agentq run <experiment> --mode full`, then `agentq compare <experiment> --summary`. Values compared: model predictions and simulated statistics against `results/paper_reported/`.

| Experiment | Paper table / figure | Values compared | Max relative difference | Wall time |
|---|---|---|---|---|
| `load_sweep` | Table 5, Fig. 7 | 90 | 3.95e-14 | 133 s |
| `fork_join` | Fig. 6 | 30 | 4.66e-15 | 104 s |
| `bottleneck` | Table 6 | 20 | 5.33e-14 | 36 s |
| `ablation` | Table 11 | 7 | 8.66e-15 | 8 s |
| `trees` | Table 7, Fig. 8 | 36 | 4.24e-14 | 84 s |
| `cv_sensitivity` | Table 8 | 12 | 3.33e-15 | 61 s |
| `burst_failure` | Table 12 | 36 | 1.34e-14 | 56 s |
| `burst_bracket` | Table 9 | 34 | 6.11e-15 | 139 s |
| `capacity` | Table 10 | 12 | 2.00e-15 | 233 s |
| `depth_sweep` | §7.9 text | 6 | 3.11e-15 | 1.3 s |
| `branching` | Table 3 | 16 | 0 | 4.8 s |
| `retries_cost` | Table 4 | statistical only | max abs P_s error 0.0014; max cost error 0.58% (paper: 0.18 pp, 0.6%) | 1.5 s |

**Result:** every comparable value reproduces to floating-point round-off. This covers 299 values across 11 experiments, including all simulator statistics, which are bit-identical under NumPy 2.5.3 (the paper used 2.4.4). `retries_cost` uses a corrected Monte Carlo (A12) and agrees statistically. The `burst_bracket` bracket held in 12/12 settings, as reported.

These runs reproduce the paper's *simulation-based* results. They do not validate the model against real agent systems.

## Failures encountered and fixes

| Failure | Detected by | Fix |
|---|---|---|
| First fidelity check hung with no output | Manual run | A stray `cat` read stdin; rewrote the check script in the scratchpad and reran: exact match |
| API returned 500 on invalid input: Pydantic error context held a `ValueError` that JSON could not serialize | `TestApi.test_validation_errors` | `exc.errors(include_url=False, include_context=False, include_input=False)` |
| 9 lint errors (import order, line length, unused loop var) | `ruff check` | Configured first-party packages; refactored lines |
| `agentq compare ... \| head` raised `BrokenPipeError`, and `--summary` was not applied (an edit missed after reformatting) | Manual run of `make compare` | Added `BrokenPipeError` handling; re-applied `--summary`; verified with `make compare` |
| Validity flags: ρ exactly 0.8 tripped "> 0.8" through round-off, while ρ = 0.9 (where the paper measured −32%) only got the weaker flag | Manual CLI/API runs | Inclusive thresholds with 1e-9 tolerance (`>= 0.8`, `>= 0.9`) |

Also found while preparing the source material (before the repository was built): the manuscript's Table 8 printed fractions instead of percentages. Fixed in `paper/main.tex`; see docs/assumptions.md, A18.

## Remaining issues and dependency concerns

- pytest, FastAPI and the Docker build are untested here (no PyPI access, no Docker daemon). CI will exercise pytest on first push.
- Requirements are pinned to the tested versions. Simulator bit-reproducibility depends on NumPy `Generator` stream stability; regression tests detect any change.

## Runtime observations

- **Model:** one prediction on the paper grid takes 0.22 s here; one capacity search (40 bisection steps) took 9.1 s.
- **Full reproduction:** the experiment run times sum to about 14 minutes; split across two processes on 2 vCPUs it finished in about 8 minutes of wall time.
