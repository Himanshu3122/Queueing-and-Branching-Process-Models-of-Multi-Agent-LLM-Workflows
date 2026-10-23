# Inference (using the model)

"Inference" here means evaluating the performance model for a configuration. Each evaluation is deterministic and takes well under a second on the paper's grid.

## CLI

```bash
# Latency distribution, success probability and cost for profile P3 at 80% LLM utilization
agentq predict --profile P3 --rho 0.8

# Same, by task rate, as JSON
agentq predict --profile P3 --rate 0.35 --json

# Max task rate for p99 <= 150 s with the paper's 0.85 derating
agentq capacity --profile P3 --p99-target 150 --derating 0.85

# One reference simulation (seeded)
agentq simulate --profile P3 --rho 0.8 --tasks 30000 --seed 0
```

`predict` also prints the stability limit, the bottleneck pool and any validity flags (see `agentq.validity`).

## Python

```python
from agentq import load_profiles, predict, rate_for_llm_utilization
from agentq.trees import predict_tree
from agentq.bursts import MMPP2, bracket
from agentq.planning import max_rate_for_p99

profiles = load_profiles("configs/profiles.yaml")
p3 = profiles["P3"]

pred = predict(p3, rate_for_llm_utilization(p3, 0.8))
print(pred.mean, pred.p95, pred.p99, pred.success_prob, pred.calls_per_success)

# Custom system: widen the tool fan-out and enlarge the LLM pool
custom = p3.with_updates(**{"workflow.fanout": 4.0, "llm.servers": 12})
print(predict(custom, 0.4).p99)

# Sub-agent trees with mean delegation m = 0.3
print(predict_tree(profiles["TREE"], 0.25, 0.3).p99)

# Bursty arrivals: Poisson (optimistic) vs quasi-static (pessimistic)
print(bracket(p3, MMPP2(rate=0.2, burst_factor=4.0)).to_dict())

# Capacity planning
print(max_rate_for_p99(p3, target_p99=150.0, derating=0.85).to_dict())
```

## HTTP API

```bash
agentq serve --port 8000           # requires the "api" extra
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' \
     -d '{"profile": "P3", "rho_llm": 0.8}'
curl -s -X POST localhost:8000/capacity -H 'content-type: application/json' \
     -d '{"profile": "P3", "target_p99": 150, "derating": 0.85}'
```

Requests accept either `"profile"` or an inline `"system"` object, and exactly one of `"task_rate"` / `"rho_llm"`. An optional `"grid": {"dt": ..., "n": ...}` is capped by `AGENTQ_MAX_GRID_POINTS`. Validation errors return 422, unknown profiles return 400.

## Interpreting outputs

- **Latency fields:** describe *successful* tasks, as in the paper.
- **`stable = false`:** some pool has ρ ≥ 1; latencies are `None`.
- **`horizon_tail_mass`:** probability mass truncated at the grid horizon. Increase `--points` or `--dt` if it is not negligible.
- **Tail accuracy:** treat p99 above ρ ≈ 0.8, or with any validity flag raised, as **optimistic**. The paper measured underestimates of up to 32% (chains at ρ = 0.9) and 40% (tool-bound fork). Use the derated capacity search.
