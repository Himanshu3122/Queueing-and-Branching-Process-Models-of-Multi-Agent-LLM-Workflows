# Paper-reported raw outputs

These JSON files are the raw outputs of the **original research scripts** from which the paper's tables and figures were generated. They are reference data, not outputs of this repository.

| File | Contents (keys) | Paper |
|---|---|---|
| `res1.json` | `e1` progeny, `e1b` extinction, `e1c` overdispersion, `est` criticality estimation, `e2` retries/cost, `e3` fork-join, `e4` load sweep | Tables 3-5, Figs. 4-7 |
| `res2.json` | `e5` Poisson model under MMPP, `e6`/`e6_sim` ablation, `e7` bottleneck, `e8` runtime, `e9` depth sweep | Tables 6, 11, 12, §7.9 |
| `res_x1.json` | `x1` sub-agent trees | Table 7, Fig. 8 |
| `res_x2.json` | `x2` LLM service CV sensitivity | Table 8 |
| `res_x3.json` | `x3` MMPP bracket | Table 9 |
| `res_x4.json` | `x4` capacity planning | Table 10 |

Original environment: Python 3.13, NumPy 2.4.4, SciPy 1.17.1, 1 vCPU. Simulation entries are `[mean, spread]` pairs, where spread is the SD for 4-seed experiments and the 95% Student-t half-width for the `x*` files.

Use `agentq compare <experiment>` to compare a `full`-mode reproduction with these files.
