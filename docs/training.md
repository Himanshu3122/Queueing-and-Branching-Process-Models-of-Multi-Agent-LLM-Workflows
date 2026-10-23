# Training and calibration

**There is no training in this project.** The paper's model is analytical: Galton–Watson branching, an absorbing retry chain, G/G/k queueing approximations and fork-join composition. It has no learned weights and therefore no checkpoints. Running it requires only parameters.

The equivalent of "training" is **calibration**: estimating the model's parameters from execution traces. The paper describes this as the main open step (Sections 9–10) but does not perform it, because no traces were available. The repository implements the estimators. They have been tested only on synthetic data (docs/assumptions.md, A1 and A13).

## Parameters to calibrate

| Parameter | Meaning | Estimator (`agentq.calibration`) |
|---|---|---|
| `failure_prob` (p) | Per-attempt failure probability | failed attempts / attempts, with binomial SE |
| `max_retries` (R) | Retry budget | maximum observed attempt index (or take it from configuration) |
| `mean_steps` (n̄) | Mean geometric step count | from successful tasks: `q = 1 - (1 - 1/mean_steps_successful)/s` |
| `fanout_kind`, `fanout` | Tool fan-out law | mean of tool calls per attempt; Poisson if the dispersion index > 0.5 |
| LLM and tool service mean and CV | Service-time distributions | MLE fits of lognormal / exponential / gamma, chosen by AIC |
| `m` (delegation) | Mean sub-agents per agent | `estimate_delegation(offspring_counts)`, with the relative SE of `E[Z]` from Eq. (5) |

Pool sizes (`servers`) come from the deployment, not from traces.

## Trace schema

One row per attempt (a Python dict, or a CSV converted to dicts):

```text
task_id, step, attempt, failed, n_tool_calls, llm_service, tool_services[]
```

Service times must exclude queueing. If the traces contain only end-to-end spans, the service-time estimates will be inflated by waiting time.

## Example (synthetic data)

```python
from agentq.config import load_profiles
from agentq.calibration import synthetic_trace, calibrate

p3 = load_profiles("configs/profiles.yaml")["P3"]
rows = synthetic_trace(p3, n_tasks=4000, seed=1)   # SYNTHETIC, generated from the model's own assumptions
est = calibrate(rows)
print(est.failure_prob, est.mean_steps, est.fanout_mean, est.llm_fit[0])
system = est.to_system(llm_servers=8, tool_servers=16)
```

`tests/unit/test_model_components.py::TestValidityAndCalibration` checks that these estimators recover known parameters.

## What calibration cannot fix

- Estimators assume the paper's structural laws: geometric steps, independent failures, Poisson or fixed fan-out. Real traces may violate them. Check goodness of fit before trusting a calibrated model.
- The paper cites measured coding-agent traces in which tool calls over one minute are about 4% of calls but 85% of tool time. Lognormal, exponential and gamma may all fit such tails poorly.
- After calibration, validate the model against held-out trace replay. The simulator currently accepts only Poisson or MMPP arrivals; trace replay is listed as future work in `IMPLEMENTATION_STATUS.md`.
