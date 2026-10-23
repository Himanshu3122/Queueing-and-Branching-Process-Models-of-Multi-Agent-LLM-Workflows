# Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `error: profiles file not found: configs/profiles.yaml` | CLI run outside the repository root | `cd` to the repo root or pass `--profiles /path/to/profiles.yaml` |
| `ModuleNotFoundError: No module named 'agentq'` | Package not installed | `pip install -e .`, or run with `PYTHONPATH=src` |
| `the API needs the 'api' extra` | `uvicorn`/`starlette` missing | `pip install -e ".[api]"` |
| `stable: false`, latencies `None` | A pool has ρ ≥ 1 | Lower the rate or add servers; `agentq predict` prints the stability limit and bottleneck |
| `horizon_tail_mass` noticeably > 0 | Grid horizon too short for the tail (long tool calls, deep workflows) | Increase `--points` or `--dt` (horizon = dt × points) |
| `max_rate` equals the lower bound (0.001) | Even tiny loads miss the derated p99 target | Target infeasible for this structure; reduce depth or fan-out or raise the target |
| `RuntimeError: no successful tasks after warm-up` | Simulation too short or failure rate too high | Increase `--tasks` |
| `agentq compare` raises "only full-mode results are comparable" | Comparing a smoke run | Run the experiment with `--mode full` first |
| Simulated numbers differ from the paper | Different seeds, run length or NumPy random streams | Use the `full` config unchanged. Seeded outputs were reproduced bit-for-bit with NumPy 2.5.3; other NumPy versions could change streams |
| `pip install` fails offline | No package index | Install `numpy scipy pydantic pyyaml` from a mirror, then `pip install -e . --no-deps --no-build-isolation` |
| Full reproduction is slow | 30,000-task simulations × seeds × settings | Run experiments in parallel shells; each experiment is independent |
| Large p99 underestimation at high load | Known model limitation (exponential wait tail; sibling batch arrivals) | Use derating (δ ≈ 0.85) and heed validity flags |
