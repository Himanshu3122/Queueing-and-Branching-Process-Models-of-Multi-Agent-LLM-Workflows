# Assumptions and engineering decisions

Every decision that is not fixed by the paper (`paper/main.tex`) is listed here. Classification:

- **A**: directly specified by the paper.
- **B**: reasonable engineering inference.
- **C**: approximation, or a discrepancy that had to be resolved.
- **D**: not implementable from the available materials.

| ID | Topic | Class | Summary |
|---|---|---|---|
| A1 | Real traces / datasets | D | The paper has none; all parameters are assumed |
| A2 | HTTP framework | B | Starlette + Pydantic instead of FastAPI |
| A3 | Test runner | B | `unittest`-style tests (pytest-compatible) |
| A4 | Erlang-C evaluation | B | Stable Erlang-B recursion instead of factorials |
| A5 | Grid in smoke mode | B | Coarser grid for speed; full mode uses the paper grid |
| A6 | "Exponential service" ablation | C | Paper's code used lognormal with CV 1; both variants provided |
| A7 | Mean-value baseline | C | Reproduces the paper's code exactly (one tool response, not success-conditioned) |
| A8 | Simulated utilization | A/C | Inherited estimator includes the warm-up period |
| A9 | Validity flags | B | Thresholds derived from the paper's reported error regimes |
| A10 | Trees at m = 0 | B | Uses the tree fixed point, which reduces to the chain model |
| A11 | Tree simulator after failure | C | Inherited: agents of failed trees keep running |
| A12 | Retry Monte Carlo | C | Single geometric draw per step (paper code used two independent draws) |
| A13 | Calibration estimators | B/D | Implemented and tested on synthetic traces only |
| A14 | Server-count search | B | Extension beyond the paper's rate search |
| A15 | Runtime numbers | C | Hardware-dependent; measured locally |
| A16 | Source-file naming | B | Report file is `PROJECT_REPORT.md`, paper copied to `paper/` |
| A17 | Report-suggested infrastructure | B | Not adopted (dependency discipline) |
| A18 | CV-table formatting bug | C | Found in the manuscript build and fixed in the paper copy |

---

### A1. No real traces or datasets (D)
- **Missing:** The paper validates the model only against its own simulator, with assumed parameters (Table 2). No agent traces, datasets or calibrated parameters exist.
- **Decision:**
  - Implement the full model, simulator and experiment protocol.
  - Implement the estimators the paper names (MLE + AIC, offspring counts for *m*) in `agentq.calibration`.
  - Provide a clearly labelled synthetic trace generator for testing them.
- **Why:** Fabricating "real" data would misrepresent the research.
- **Change later:** Feed real attempt rows into `agentq.calibration.calibrate` (schema in its module docstring).

### A2. Starlette instead of FastAPI (B)
- **Missing:** FastAPI could not be installed in the build environment (PyPI was unreachable). Starlette, uvicorn and Pydantic were available.
- **Decision:** The API is written with Starlette (the ASGI toolkit FastAPI is built on) plus Pydantic request models. It was started and exercised locally.
- **Why:** Prefer code that actually ran over an untested FastAPI version. The footprint is also smaller.
- **Change later:** Wrapping the same handlers in FastAPI routes is mechanical. The request models in `agentq/api/app.py` are already Pydantic.

### A3. `unittest`-style tests (B)
- **Missing:** pytest could not be installed locally.
- **Decision:** Tests are `unittest.TestCase` classes, which pytest also collects. `make test` uses `unittest`, and CI uses pytest.
- **Change later:** None needed.

### A4. Erlang-C via the Erlang-B recursion (B)
- **Missing:** The paper writes Eq. (11) with factorials, which overflow for large *k*.
- **Decision:** Use the algebraically identical recursion. Predictions match the paper's stored outputs to about 1e-14 relative (verified in `tests/integration/test_reference_regression.py`).

### A5. Coarser grid in smoke mode (B)
- **Missing:** The paper fixes `dt = 0.02 s` and `30,000` points (600 s horizon) for the model.
- **Decision:** `full` mode uses exactly this grid. `smoke` mode and the unit tests use `dt = 0.05 s`, `6,000` points (300 s horizon) for speed.
- **Additional output:** `Prediction.horizon_tail_mass` reports mass truncated at the horizon, so users can detect when the grid is too short (e.g., minute-scale tool calls).

### A6. The paper's "Exponential service (M/M/k)" ablation (C)
- **Discrepancy:** The paper's Table 11 row is labelled exponential service. The original script computed it as *lognormal with CV = 1* for both pools, not as exponential.
- **Decision:** The `ablation` experiment reports both:
  - `lognormal_cv1_paper_exponential_ablation` (reproduces the paper's number);
  - `true_exponential_service` (`ModelOptions(service_dist="exponential")`).
- **Why:** This keeps reproduction exact while making the true ablation available. The paper's conclusion ("changes errors by under one point because CV 0.9 is close to 1") should be re-checked against the true variant.

### A7. Mean-value baseline (C)
- **Paper:** "sums mean waits and service times as a Little's-law style capacity calculation would".
- **Code behavior:** The original code multiplies `nu * a` by the per-attempt sum of mean LLM and tool waits and services. It counts one tool response (not the fan-out maximum) and does not condition on success. `mean_value_baseline` reproduces exactly this.
- **Change later:** A stronger baseline would use `E[K]` or the expected maximum of `K` responses.

### A8. Simulated utilization (A/C)
The simulator reports `busy time / (servers × last arrival time)`, including the warm-up period, as the paper's code did. It is a diagnostic only; utilization targets are set analytically (Eq. (9)).

### A9. Validity-flag thresholds (B)
- **Source:** The paper reports where the model is inaccurate but defines no runtime checks.
- **Decision:** `agentq.validity` raises a flag for each condition below, citing the evidence:
  - utilization > 0.8 / > 0.9 (Table 5);
  - a tool-bound pool (Table 6);
  - LLM CV outside [0.5, 1.5] (Table 8);
  - tool CV > 1.5, which is untested (Discussion);
  - delegation m > 0.5, which is untested (Table 7);
  - arrival SCV > 1.2 (Table 12).
- **Change later:** Edit the thresholds in `validity.py` after calibrating on traces.

### A10. Trees with m = 0 (B)
The paper's script used the chain model for `m = 0`. Here the tree fixed point is used for all *m*. At `m = 0` it reduces to the chain model, which a unit test checks to 1e-6.

### A11. Tree simulator behaviour after a failure (C)
Inherited from the paper's code: once any agent exhausts its retries, the tree is recorded as failed. Its other agents keep running and consuming capacity, and the failed agent's parent is never released. This affects pool load slightly but not the latency of *successful* trees. It is kept for exact reproduction and documented in `sim/tree.py`.

### A12. Monte Carlo for retries and cost (C)
- **Original code:** Drew a step's attempt count and its failure flag from two independent geometric variates. This is statistically equivalent in expectation but not a faithful simulation of one step.
- **Decision:** Use one draw. Expectations are unchanged, so the Table 4 conclusions are unaffected, but the numbers are not bit-identical to the paper's Monte Carlo.

### A13. Calibration estimators (B/D)
- **Step-count estimate:** Successful tasks have geometric step counts with parameter `1 - (1 - q)s` (the success tilt in Eq. (18)). Hence `q = 1 - (1 - 1/mean_successful_steps)/s`.
- **Fan-out law:** Poisson if the variance/mean of tool calls per attempt exceeds 0.5, otherwise fixed.
- **Service times:** Lognormal, exponential and gamma are fitted by MLE and chosen by AIC.
- **Tested on:** Synthetic traces only. Real traces will need richer families (e.g., heavy-tailed mixtures) given the measured tool tails cited in the paper.

### A14. Minimum server search (B)
`planning.min_llm_servers_for_rate` is an engineering extension. The paper evaluates only the rate search of Eq. (27).

### A15. Runtime measurements (C)
The paper reports 0.52 s per model evaluation and 2.2 s per 30,000-task simulation on its machine. The `runtime` experiment re-measures on the local machine. These numbers are not comparable across hardware.

### A16. Source-file naming (B)
The engineering report in the source directory is `PROJECT_REPORT.md` (upper case), not `project_report.md`. The paper is `journal_version_final/main.tex`. An unmodified copy of the paper and bibliography is included in `paper/` for traceability.

### A17. Infrastructure suggested by the report but not adopted (B)
`PROJECT_REPORT.md` suggests PostgreSQL, OpenTelemetry ingestion, MLflow/DVC, Ray and a web dashboard. None is needed to implement or reproduce the paper, so none is included.

- **Results:** Stored as JSON with provenance.
- **Container:** A single Dockerfile covers the one service.
- **docker-compose:** Not provided, because there is only one service.

### A18. Manuscript table bug (C)
The paper build originally printed Table 8's error columns as fractions (e.g. "−0.1" for −5.4%). The prose had the correct values. This was found and fixed in the source manuscript before the repository was built, and `paper/main.tex` is the corrected version. The `cv_sensitivity` experiment output contains the correct percentages.
