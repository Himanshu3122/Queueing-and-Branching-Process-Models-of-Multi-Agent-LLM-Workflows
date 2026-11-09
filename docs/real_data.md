# Real-data experiments

The paper's experiments use synthetic workloads only. This page documents the first experiments on
**real, public data**. They deliberately start with two small datasets (under 2 MB in total) so the
whole pipeline runs on a laptop.

```bash
pip install -e ".[real]"                          # pyarrow + tiktoken
python scripts/fetch_datasets.py                  # downloads to data/raw and checks SHA-256
python -m agentq run real_calibration --mode full --out results/real
python -m agentq run real_validation  --mode full --out results/real    # ~20 min
```

`--mode smoke` runs a reduced version in about 10 s. Results are written to
`results/real/full/real_calibration.json` and `results/real/full/real_validation.json`.

## Datasets

| Dataset | What it contains | Used for | Licence |
|---|---|---|---|
| **AgentInstruct** (Zeng et al., 2023, *AgentTuning*), `zai-org/AgentInstruct` on Hugging Face | 1,866 GPT-4 agent trajectories on AgentBench tasks. We use the 1,744 multi-step ones: OS (195), DB (538), ALFWorld (336), WebShop (351), knowledge graph KG (324). Mind2Web samples are single decisions and are excluded. | Workflow structure: steps per task, failed actions and retries, tool calls per step, and per-call context and output tokens (GPT-4 `cl100k_base` tokenizer). | See the dataset card |
| **Azure LLM inference trace 2023, conversation** (Patel et al., ISCA 2024, *Splitwise*), `Azure/AzurePublicDataset` | 19,366 requests to a production LLM conversation service over one hour, with timestamps and token counts. | Task arrival process. Each request is treated as one task arrival, and the trace is rescaled in time to reach the target load. | CC-BY 4.0 |
| **Mooncake traces** (Qin et al., FAST 2025, Best Paper), `kvcache-ai/Mooncake/FAST25-release` | One hour each of Kimi's **tool/agent** workload (23,608 requests) and **conversation** workload (12,031 requests): timestamps, input and output tokens, and 512-token prefix-block hashes. | Task arrival process from a real agent workload; token lengths of real tool/agent calls. | Apache-2.0 |

The data files are not committed. `scripts/fetch_datasets.py` downloads them and checks the
pinned checksums in `src/agentq/datasets/fetch.py`.

## Measured vs assumed

| Quantity | Source |
|---|---|
| Steps per task, its distribution | **measured** (AgentInstruct) |
| Failed attempts, retries, clustering of failures | **measured**. An attempt fails when the next observation is an environment error: SQL error code, shell error, ALFWorld `Nothing happens.`, KG `cannot be executed` / `No executable function found`. |
| Tool calls per attempt | **measured**. ReAct agents issue 0 or 1 tool call: 0 for the final answer or an unparsable action. |
| Context and output tokens per LLM call | **measured** (exact GPT-4 token counts) |
| Task arrival times | **measured** (Azure trace, rescaled to the target rate) |
| LLM service time per call | **derived** from measured tokens by an **assumed** linear latency model: 0.25 s + 0.2 ms × context tokens + 25 ms × output tokens. A decode-speed sensitivity check (12.5 and 50 ms/token) is included. |
| Tool service time | **assumed**: lognormal, mean 1.0 s, CV 1.5 (the paper's default; not recorded in either dataset) |
| Pool sizes | **assumed**: `k_L = 8`, `k_T = 16` (as in the paper's profiles) |

### What these data cannot identify

* **Task-level failure and the retry cap R.** AgentInstruct keeps only trajectories that solved the
  task, so every task succeeds and `R` is set to the largest number of retries observed. The failure
  probability `p` is a rate of failed *actions* within successful runs.
* **Latency.** Neither dataset has per-call timings, so the experiments validate the model against a
  *replay* of real workflow structure through simulated pools. They do not validate it against
  measured end-to-end latency. That needs traces from an instrumented agent deployment (the paper's
  stated next step).
* **Joint workflow and arrival data.** Workflows and arrivals come from different systems. Pairing
  them assumes the conversation service's request pattern is a reasonable stand-in for agent task
  arrivals.

## Experiments

**`real_calibration`** fits the model to each environment with
`agentq.calibration.calibrate` (and to the pooled `mixed` workload). It then tests each assumption:
the geometric step count (variance ratio, KS distance, p99), independence of failures
(P(fail | previous attempt failed) vs `p`), the service-time family (AIC over lognormal, gamma and
exponential), the dependence of service time on position in the task, and Poisson arrivals
(interarrival SCV, lag-1 correlation, index of dispersion of counts at 1–300 s windows).

**`real_validation`** compares the calibrated model (revised numerics) with three references at
`rho_L` in {0.3, 0.5, 0.7, 0.8, 0.9}, using 4 seeds of 19,366 tasks each:

1. `sim_model_assumptions`: the paper's simulator with the calibrated parameters. The gap here is
   pure approximation error, as in the paper.
2. `replay_poisson`: the real trajectories replayed exactly, with Poisson arrivals. The added gap
   is **workflow-structure mismatch**.
3. `replay_azure`: the same replay driven by the real Azure timestamps. Each seed rotates the trace
   to a different start point. The added gap is **arrival burstiness**.

## Results summary

Full mode: 6 workloads × 5 loads × 3 references × 4 seeds (21 minutes on a laptop).

### Assumption tests (`real_calibration`)

| Workload | Tasks | `p` | Steps mean | Step variance (geometric) | KS to geometric | P(fail after fail) | LLM `E[S]` (CV) |
|---|---:|---:|---:|---:|---:|---:|---:|
| OS | 195 | 0.027 | 3.75 | 1.7 (10.3) | 0.28 | 0.20 | 1.72 s (0.39) |
| DB | 538 | 0.029 | 2.00 | 0.08 (2.0) | 0.48 | 0.41 | 2.81 s (0.87) |
| ALFWorld | 336 | 0.006 | 13.4 | 72.7 (167) | 0.20 | 0.04 | 0.76 s (0.50) |
| WebShop | 351 | 0.000 | 3.68 | 0.72 (9.9) | 0.47 | — | 1.42 s (0.17) |
| KG | 324 | 0.069 | 13.1 | 2.3 (158) | 0.51 | 0.69 | 1.61 s (0.20) |
| Mixed | 1,744 | 0.032 | 6.79 | 40.7 (39.4) | 0.14 | 0.59 | 1.39 s (0.72) |

* **Geometric step counts are rejected for every single environment.** Real tasks are far more
  regular than geometric (KG: variance 2.3 against 158). Only the heterogeneous mix happens to
  match the geometric variance.
* **Failures are not independent.** Except in ALFWorld, a failed action is followed by another
  failure 20–69% of the time, against a base rate of 3–7%.
* **Lognormal is the AIC-best service-time family** in every workload, supporting the model's
  choice.
* **Azure arrivals are bursty on long time scales.** Interarrival SCV is only 1.20 and lag-1
  correlation 0.06 (Poisson-like at short range), but the index of dispersion of counts is 1.4,
  3.9, 16 and 56 at 1, 10, 60 and 300 s windows (Poisson ≈ 1). The rate exceeds 1.25× its hourly
  mean in 18% of 5-minute windows (peak 1.35×).

### Model error (`real_validation`)

Relative error of the model (model / reference − 1) for p99, in percent, with the mean error in
brackets, at `rho_L` = 0.5 / 0.8:

| Workload | vs own-assumption sim | vs replay, Poisson arrivals | vs replay, Azure arrivals |
|---|---:|---:|---:|
| OS | −1 (−1) / −8 (−5) | +84 (+10) / +60 (+3) | +80 (+9) / −92 (−88) |
| DB | +1 (+1) / −4 (0) | +34 (+14) / +34 (+8) | +33 (+12) / −90 (−89) |
| ALFWorld | 0 (0) / −6 (−4) | +53 (0) / +44 (−3) | +52 (−1) / −91 (−83) |
| WebShop | −1 (0) / −9 (−5) | +78 (0) / +68 (−5) | +76 (−1) / −90 (−87) |
| KG | −1 (0) / −13 (−8) | +145 (+10) / +108 (+2) | +140 (+8) / −92 (−89) |
| Mixed | −1 (0) / −9 (−4) | +23 (+7) / +12 (+4) | +21 (+5) / −93 (−83) |

All five loads (0.3–0.9), with 95% confidence intervals, are in
`results/real/full/real_validation.json`.

Reading the three columns as a decomposition:

1. **Approximation error** (column 1) reproduces the paper's synthetic finding with
   real-calibrated parameters: within 1% at `rho_L ≤ 0.5`, a p99 underestimate of 4–13% at 0.8
   and 12–31% at 0.9.
2. **Workflow structure** (column 2): **the mean is predicted well** (within ±15% everywhere,
   within ±5% for ALFWorld, WebShop and the mix), but **the tail is overestimated by 12–146%**.
   The geometric step count produces far more very long tasks than real agents do. The error is
   smallest for the mixed workload, whose step-count variance happens to be geometric-like.
3. **Real arrivals** (column 3): up to `rho_L = 0.5` the Azure trace changes almost nothing. At
   0.7 the model already underestimates (mean −20 to −28%). At 0.8–0.9 the real rate surges
   overload the pool and the model underestimates latency by roughly an order of magnitude (p99
   −90 to −97%, with wide seed-to-seed spread because the result depends on where the surges
   fall). This confirms on a real trace the paper's synthetic MMPP finding: a stationary Poisson
   model should not be used for planning above about `rho_L = 0.6` on bursty traffic. The 15%
   derating rule (target `rho_L` ≈ 0.68) is not enough for this trace.

**Sensitivity to the assumed decode speed** (`rho_L = 0.8`, 12.5 vs 50 ms per token): the
conclusions do not change. The KG p99 error against the Poisson replay is +122% and +91%, the mixed
workload's is +8% and +12%, and every Azure-replay p99 error stays between −90% and −94%.

## Dataset 2: Mooncake arrival traces (FAST 2025)

```bash
python -m agentq run real_arrival_traces       --mode full --out results/real   # 2 s
python -m agentq run real_validation_mooncake  --mode full --out results/real   # 4 min
```

The validation repeats `real_validation`, but drives the replay with the Mooncake tool/agent and
conversation timestamps. Each run uses 12,031 tasks, the length of the conversation trace. The
tool/agent trace therefore contributes a 30-minute window per seed, rotated across seeds.

### Data findings

| Trace | Rate | IDC 60 s / 300 s | Peak / mean (5 min) | 5-min windows above 1.25 × mean |
|---|---:|---:|---:|---:|
| Azure conversation | 5.5 /s | 16 / 56 | 1.35 | 18 % |
| Mooncake tool/agent | 6.7 /s | 3.8 / 10.5 | 1.07 | 0 % |
| Mooncake conversation | 3.4 /s | 2.9 / 7.3 | 1.09 | 0 % |

* **Timestamp resolution.** Mooncake timestamps are binned to 3-second slots: there are 1,180
  distinct values in an hour, with about 20 tool/agent requests per value. Requests are spread
  evenly within their slot, so statistics below about 3 s (interarrival SCV 0.09, IDC 0.5 at 1 s)
  are artifacts and are not used. At 10 s and longer the statistics are exact.
* **Both Mooncake workloads are burstier than Poisson but much smoother than Azure** over minutes.
  The rate never exceeds 1.25 × the hourly mean in a 5-minute window.
* **Real tool/agent calls have about 10× longer prompts than AgentInstruct's**: mean 8,596 input
  tokens (CV 1.28) against AgentInstruct medians of 400–1,200. Output is 182 tokens (CV 1.33).
  55 % of prompt blocks repeat an earlier prefix (KV-cache reuse), which the FIFO pool model does not
  represent. Lognormal is again the AIC-best family for the implied service time of tool/agent
  calls; gamma fits best for conversation.
* **Sessions cannot be reconstructed.** Chaining requests by prefix hashes links only 17 % of
  tool/agent requests, with a median gap of 117 s, because the traces are sampled. Workflow structure
  therefore still comes from AgentInstruct.

### Model error with Mooncake arrivals

p99 error in percent (mean error in brackets). The Azure column is from `real_validation`, which
used 19,366 tasks per run.

| Workload | `rho_L` | Poisson replay | Mooncake tool/agent | Mooncake conversation | Azure |
|---|---:|---:|---:|---:|---:|
| OS | 0.7 / 0.8 / 0.9 | +77 / +66 / +35 | +64 (+6) / +23 (−5) / −15 (−28) | +63 / +33 / −9 | −39 / −92 / −96 |
| DB | 0.7 / 0.8 / 0.9 | +39 / +41 / +43 | +30 (+9) / +11 (−3) / −17 (−33) | +32 / +19 / −6 | −35 / −90 / −95 |
| ALFWorld | 0.7 / 0.8 / 0.9 | +51 / +45 / +18 | +45 (−3) / +20 (−8) / −17 (−25) | +48 / +28 / −16 | −35 / −91 / −96 |
| WebShop | 0.7 / 0.8 / 0.9 | +76 / +67 / +38 | +67 (−3) / +34 (−11) / −4 (−32) | +64 / +42 / +5 | −24 / −90 / −96 |
| KG | 0.7 / 0.8 / 0.9 | +133 / +106 / +46 | +103 (+4) / +38 (−8) / −12 (−37) | +102 / +47 / −1 | −28 / −92 / −96 |
| Mixed | 0.7 / 0.8 / 0.9 | +18 / +10 / −24 | +16 (+4) / −3 (−2) / −32 (−17) | +16 / +1 / −37 | −38 / −93 / −97 |

At `rho_L` ≤ 0.5 all three real traces give the same result as Poisson arrivals, within 3 points.

**What changes with this dataset:**

1. **The tenfold collapse seen with Azure is not universal.** It comes from Azure's rate surges.
   Under the real agent workload (Mooncake tool/agent), the model's mean stays within ±11 % up to
   `rho_L = 0.8` for every workload. On the mixed workload p99 is within 3 % at 0.8.
2. **Moderate bursts partly cancel the geometric-tail overestimate.** The +20 to +106 % p99
   overestimate under Poisson replay at `rho_L = 0.8` shrinks to −3 to +38 % under Mooncake.
   The model is right here partly for the wrong reason: two errors of opposite sign offset.
3. **At `rho_L = 0.9` the model underestimates under every real trace.** Mean errors range from −17
   to −37 % under the Mooncake tool/agent trace.
4. **Practical rule from two real arrival traces:** the stationary model is usable for planning
   up to about `rho_L = 0.7` on any of the three traces. It is usable up to 0.8 only when the
   5-minute rate rarely exceeds 1.25 × its mean, so check `rate_surges` on your own trace before
   relying on it.

## Dataset 3: BFCL tool fan-out (ICML 2025)

```bash
python -m agentq run real_fanout --mode full --out results/real   # 12 min
```

**Data.** We use the Berkeley Function Calling Leaderboard (Patil et al., ICML 2025; Apache-2.0).
The fan-out `K` of a query is the number of calls in its ground-truth answer. Only **BFCL-Live** is
used for frequencies: it holds real user queries contributed by enterprises and open-source users,
deduplicated and filtered by BFCL. The author-designed parallel categories are reported for
reference only, because their mix of fan-outs is a design choice.

| Fan-out law | Queries | `E[K]` | Var / mean | P(K ≥ 2) | AIC Poisson vs empirical |
|---|---:|---:|---:|---:|---|
| BFCL-Live, all relevant queries | 1,351 | 1.04 | 0.07 | 3.0 % | 2,794 vs 425 |
| BFCL-Live, parallel queries only | 40 | 2.35 | 0.34 | 100 % (82.5 % have K = 2, max 6) | 121 vs 63 |
| Designed parallel (reference) | 400 | 2.86 | 0.29 | 100 % (max 8) | — |

* **Real queries rarely fork.** 97 % issue exactly one call, and K ≥ 6 occurs in 0.07 % of queries.
* **Neither of the model's fan-out laws fits.** Real K is strongly underdispersed: Poisson is
  rejected by a wide AIC margin, and the fixed-K rule misstates the mean (round(2.35) = 2).

**Experiment.** The fork-join part is tested in isolation. The reference simulation follows every
model assumption (P3 profile: geometric steps, `p = 0.08`, `R = 1`, lognormal service, Poisson
arrivals), but draws each attempt's K from the measured pmf. All model variants run at the same
arrival rate, which is set from the true `E[K]`. There are two pool regimes. In the LLM-bound
regime `k_L = 8`, `k_T = 16`. In the tool-bound regime `k_L = 32` and `k_T` is sized so that tool
utilization is 1.25× LLM utilization (`k_T = 9` and 20). We use 4 seeds × 30,000 tasks.

p99 error in percent (mean error in brackets), tool-bound regime:

| Law | `rho_T` | Poisson K | Fixed K = round(E[K]) | Fixed + batch `c_a²=K` | Poisson + measured `c_a² = E[K²]/E[K]` |
|---|---:|---:|---:|---:|---:|
| Live, all | 0.7 | −4.6 (−4.4) | −1.4 (−0.7) | −1.4 (−0.7) | −4.5 (−4.3) |
| Live, all | 0.9 | −3.0 (−1.0) | −4.5 (−1.8) | −4.5 (−1.8) | −2.3 (−0.2) |
| Live, parallel | 0.7 | −3.7 (−3.7) | −4.6 (−4.4) | −4.5 (−4.3) | −3.3 (−3.3) |
| Live, parallel | 0.9 | −8.1 (−5.8) | −16.5 (−14.7) | −16.0 (−14.2) | **−2.2 (+0.6)** |

In the LLM-bound regime every variant is within 5 % up to `rho_L = 0.7`. At 0.8 to 0.9 the errors
are −9 to −22 % p99, the same as the paper's general approximation error at that LLM load.

**Findings.**

1. **With real fan-outs, fork-join is a small error source.** Every variant is within 8 % p99 up to
   `rho_T = 0.8`, and the best variant is within 4 % everywhere in the tool-bound regime. This is far
   from the paper's synthetic worst case (fixed K = 6, a −39.6 % p99 underestimate), because real
   queries almost never fork six ways.
2. **Rounding the mean is the main risk.** With forking queries (`E[K] = 2.35`), fixed K = 2
   understates tool load and underestimates p99 by 16.5 % at `rho_T = 0.9`. The paper's batch
   correction does not fix this, because the mean itself is wrong.
3. **Recommended calibration:** use a Poisson K with the true mean, together with the *measured*
   `E[K²]/E[K]` as the tool-pool `c_a²` (already supported through `ModelOptions(tool_ca2=<float>)`).
   This variant was best or tied for best at every tool-bound point (worst p99 error −4.5 %).
4. **Caveat:** the parallel-only law rests on only 40 real queries.

## Dataset 4: Who&When multi-agent failures (ICML 2025)

```bash
python -m agentq run real_multiagent --mode full --out results/real   # 6 s
```

**Data.** Who&When (Zhang et al., ICML 2025, spotlight; MIT) is fetched from
`ag2ai/Agents_Failure_Attribution` pinned to commit `1bcbbd9`. It holds 184 multi-agent runs, and
**every run failed its task**. Each run is annotated with the agent and step whose error decided the
failure. There are two systems:

* **Algorithm-Generated (126 runs):** CaptainAgent expert teams. Code executions on a
  `Computer_terminal` report an exit code, so tool failures are observable. Histories are
  truncated at 10 messages, so their run lengths are not natural.
* **Hand-Crafted (58 runs):** Magentic-One, in which an Orchestrator delegates to WebSurfer,
  FileSurfer, Assistant (coder) and ComputerTerminal.

There are no timestamps, so this dataset tests structural and failure assumptions only. Because
every run failed, nothing that depends on successful runs can be estimated.

**1. Failure independence (CaptainAgent code executions).** 236 executions fail at a rate of 38.1 %
(95 % CI 32–44 %). After a failed execution, the next fails 26.7 % of the time (n = 60, CI 17–39 %).
After a successful one, the next fails 30.7 % of the time (n = 75, CI 21–42 %). Fisher's exact test
gives p = 0.70, so **there is no evidence of clustering**. This differs from AgentInstruct, where a
failed action was followed by another failure 20–69 % of the time against base rates of 3–7 %. Whether
failures cluster therefore depends on the kind of failure. Re-planning after an environment error
(AgentInstruct) clusters; code-execution failures inside a failing team do not.

**2. Delegation is a one-level star, not a Galton–Watson tree (Magentic-One).** The Orchestrator
delegates 11.9 times per task (variance 90.4, dispersion 7.6, range 0–30). 87 % of delegations go
to WebSurfer. Every delegation produces exactly one sub-agent turn (652 of 652), and sub-agents never
delegate, so trees have depth 1. Applying the paper's estimator to the real offspring counts gives
`m̂ = 0.92 ± 0.15`. The relative standard error of the implied tree size is 194 %, the
near-critical regime in which the paper warns that Eq. (5) makes `E[Z]` unidentifiable. The implied
Borel law matches the mean by construction but not the shape:

| Tree size (agent invocations) | Real (Magentic-One) | Galton–Watson, Poisson `m̂ = 0.92` |
|---|---:|---:|
| Mean | 12.9 | 12.9 |
| Variance | 90 | 1,971 |
| Median | 9 | 2 |
| p99 | 31 | 199 |
| P(no delegation) | 1.7 % | 39.8 % |

The GW model would predict that 40 % of tasks never delegate and that 1 % spawn 199 or more agents.
Real orchestrators almost always delegate, and they stop at about 30.

**3. Failed runs do not stop at the failure.** The decisive error falls at a median of 33–34 % of
the run (interquartile range 11–56 % for CaptainAgent, 18–77 % for Magentic-One). **Half of a failed
run's agent turns come after the decisive error**: 50 % on average for CaptainAgent and 54 % for
Magentic-One (median 66 %). The model ends a failed task at the step whose retries ran out, so it
counts only the calls up to the failure. Real failures are silent and the run continues, so the model
undercounts the LLM calls spent on failed tasks. For the same success probability, it therefore
understates the cost per successful task.

**What this means for the model.**

* The sub-agent tree model (homogeneous Poisson delegation with `m < 1`) does not describe
  orchestrator-worker systems. A two-type model (an orchestrator with an overdispersed delegation
  count, workers that are leaves) matches the data structurally, and its load follows directly
  from `E[N]`.
* Retries with independent failures are adequate for tool-execution errors (this dataset) but not
  for agent re-planning (AgentInstruct).
* Cost per success needs a term for undetected failures, that is, calls spent after the decisive
  error.

## Dataset 5: SWE-Gym coding-agent runs (ICML 2025)

```bash
python -m agentq run real_swegym            --mode full --out results/real   # 1 s (+52 s one-off extraction)
python -m agentq run real_validation_swegym --mode full --out results/real   # ~7 min
```

**Data.** We use the SWE-Gym sampled OpenHands trajectories (Pan et al., ICML 2025), pinned to Hugging
Face revision `baf3a4e`: three parquet shards, 301 MB, with checksums verified. The card states no
licence; the companion SFT set is MIT. There are **6,055 runs on 2,438 real GitHub issues**, from
GPT-4o (5,826 runs at temperatures 0–1, with iteration caps of 30 or 50) and Claude 3.5 Sonnet
(229 runs, cap 50). **Both outcomes are present**: `resolved` means the hidden tests pass. This is the
first dataset here in which success, failure and cost come from one system. A one-off extraction
writes a compact record per run (`data/processed/swegym_records.jsonl.gz`). One assistant message
counts as one LLM call. Tokens are counted with `o200k_base`, GPT-4o's tokenizer, for both models.
There are no timings.

Failed attempts use two definitions. **Strict** counts tool-usage errors: an editor `ERROR:`, or a
bash exit code of 2, 126 or 127. **Broad** also counts any non-zero exit, which includes the
reproduction scripts that agents run on purpose to watch the bug fail.

### Outcomes and cost

| | All | GPT-4o | Claude 3.5 Sonnet |
|---|---:|---:|---:|
| Resolved | 8.1 % | 7.3 % | 29.3 % |
| Silent failure (wrong patch) | 52.6 % | 52.0 % | 67.2 % |
| Detected failure (no patch; mostly aborted) | 39.3 % | 40.7 % | 3.5 % |
| LLM calls per run: resolved / silent / detected | 19.2 / 28.1 / 6.5 | 18.9 / 28.0 / 6.4 | 21.5 / 30.8 / 28.2 |
| Share of all LLM calls spent on silent failures | **78 %** | 79 % | 74 % |
| Model `C_s = n̄/(1−p)` (Eq. 8), calibrated on resolved runs | 19.2 | 18.9 | 21.5 |
| **Measured calls per resolved task** | **233** | **255** | **96** |
| Understatement factor | **12.1×** | 13.5× | 4.4× |

* **The cost-per-success theorem fails in practice, and the cause can be identified.** Calibrated
  on resolved runs, the model's only failure mechanism is a step that exhausts its retries. That
  gives `P_s ≈ 1.000` against a measured 8.1 %, and it understates the cost per success by 4–14×.
  The theorem `C_s = n̄/(1−p)` is exact under the model's assumptions. What breaks is that **most
  real failures are not caused by tool errors**: they are wrong patches submitted with `finish`.
* **Detected failures are cheap; silent failures are expensive.** Aborted runs stop after a median
  of 4 calls, when OpenHands' stuck-loop detector fires after about 4 identical actions (median
  tail of 4). This mechanism does behave like the model's retry cap. Silent failures run longer
  than successes (28 vs 19 calls) and consume 78 % of all LLM calls.
* **Mapping to the Who&When finding:** failed runs continue past their decisive error. This dataset
  quantifies the cost of that at scale.

### Structure

* **Step counts are again far less variable than geometric.** For resolved runs the mean is 17.7
  steps, with variance 81 against 296 for a geometric distribution (KS distance 0.33). This is the
  third dataset to show it, after AgentInstruct and Who&When/Magentic-One.
* **Fan-out per call:** 86.5 % of calls make one tool call, 11.6 % make none and 1.9 % make two or
  more (variance/mean 0.18). This agrees with BFCL-Live: real agents rarely fork.
* **Failure clustering depends on the failure type, which reconciles the earlier datasets.**
  Strict tool-usage errors cluster strongly. P(fail | previous failed) is 0.26 against 0.076 after
  a success (odds ratio 4.3; Fisher p below floating-point resolution), matching AgentInstruct. Under the broad definition,
  dominated by script-run exit codes, clustering almost vanishes (0.194 vs 0.181, odds ratio 1.09),
  matching Who&When's code executions.
* **Context grows about 16×** within a run, from 907 to 14,642 mean prompt tokens between the first
  and last call (median 12,110 tokens per call, comparable with Mooncake's tool/agent trace). LLM
  service time is therefore strongly step-dependent, while the model assumes i.i.d. service.

### Latency of resolved tasks (`real_validation_swegym`)

Each workload is calibrated as in `real_validation`, from all of its runs, with the same assumed token
latency model. That gives a mean LLM service time of 6.2–9.5 s, driven by the long contexts. All
runs are then replayed exactly, with their real call sequences, fan-outs and token-derived service
times, through `k_L = 8` and `k_T = 16` pools. Unresolved runs load the pools but are not timed. We
use 3 seeds × 12,031 tasks, and the model grid is 3,000 s because coding runs last minutes.

p99 error in percent (mean error in brackets):

| Workload | `rho_L` | vs own-assumption sim | Poisson replay | Mooncake tool/agent replay | Azure replay |
|---|---:|---:|---:|---:|---:|
| All runs | 0.5 | +1 (0) | +11 (+8) | +10 (+9) | +12 (+7) |
| All runs | 0.8 | −11 (−2) | +1 (0) | −2 (−5) | −76 (−63) |
| All runs | 0.9 | −29 (−8) | −2 (−2) | −33 (−27) | −89 (−83) |
| GPT-4o | 0.5 | −2 (0) | +11 (+13) | +17 (+15) | +15 (+13) |
| GPT-4o | 0.8 | −11 (−3) | +14 (+7) | +8 (+2) | −68 (−54) |
| Claude 3.5 | 0.5 | −1 (−1) | +72 (+50) | +72 (+50) | +71 (+47) |
| Claude 3.5 | 0.8 | −12 (−5) | +65 (+43) | +24 (+30) | −76 (−61) |

All loads, with confidence intervals, are in `results/real/full/real_validation_swegym.json`. Azure
replays at `rho_L ≥ 0.8` have very wide CIs, which is the surge behaviour seen in Dataset 1.

* **For GPT-4o, and for all runs together, the model predicts the latency of successful tasks within
  about ±15 %** under Poisson and Mooncake arrivals up to `rho_L = 0.8`, on real coding-agent
  structure. This is the best agreement of any real workload so far. As with the AgentInstruct
  `mixed` workload, the heterogeneous mix of runs makes the geometric step law a reasonable
  approximation.
* **For Claude, the model overestimates by about 50 % (mean) and 70 % (p99).** The cause is a new
  mechanism: **success is correlated with run length.** The model takes its step count from all runs
  and uses it for the timed successful runs as well; it conditions on success only through the
  retry tilt. Claude's resolved runs are much shorter than its failed ones (21.5 vs 30.8 calls),
  while GPT-4o's are close to its average (18.9 vs 18.5 for all runs). A step law conditioned on the
  outcome, fitted to successful runs only, would remove this bias.
* **The arrival effects match Datasets 1 and 2.** Mooncake arrivals are harmless up to 0.8 and give
  underestimates at 0.9. The Azure surges collapse the prediction at 0.8 and above.

## Dataset 6: BurstGPT, months of Azure OpenAI traffic (KDD 2025)

```bash
python -m agentq run real_burstgpt --mode full --out results/real   # 25 s
```

**Data.** BurstGPT (Wang et al., KDD 2025; CC-BY-4.0), release `v2.0`, records regional Azure OpenAI
GPT-3.5 ("ChatGPT") and GPT-4 traffic. We use **part 1** (61 days, 1.43M requests, 52 MB) and
**part 3** (110 days, 5.34M requests, 232 MB). Part 2 adds the middle months but no new columns.
Part 3 adds two columns. **`Elapsed time`** is the *measured* time from submission to the complete
response, in whole seconds. **`Session ID`** identifies conversation sessions. A response of 0
tokens marks a failed request. Timestamps are whole seconds and are spread evenly within each
second.

**Measured latency: a data-quality caveat.** For API-mode requests, `Elapsed time` is 0 for 58 % of
successful requests, but never for responses of 200 or more tokens, and on 5 of the 110 days more than
90 % of API requests show 0. We treat this as a logging artifact, and the latency fit uses only
**conversation-mode requests** (157k ChatGPT, 74k GPT-4).

### 1. Rate surges are the norm in production LLM traffic

| Trace | 5-min windows above 1.25 × their hour's mean | Hours with at least one such window | p99 window / hour ratio |
|---|---:|---:|---:|
| BurstGPT part 1 (61 days) | 26 % | 96 % | 5.2 |
| BurstGPT part 3 (110 days) | 21 % | 83 % | 4.5 |
| Azure 2023 (1 hour) | 18 % | — | 1.35 (peak) |
| Mooncake tool/agent (1 hour) | 0 % | — | 1.07 (peak) |

Measured against the hour's own mean, which is the fair comparison with a one-hour trace,
**more than 80 % of hours contain a surge** strong enough to overload a pool sized for `rho_L = 0.8`. *(Dataset 8 qualifies this: it holds for this low-volume, single-customer stream, while
high-volume services are far smoother within the hour.)*
The Azure 2023 hour is typical, and the Mooncake hour is unusually smooth. The index of dispersion
of counts reaches 5,000–10,000 at one-hour windows, and the diurnal peak-to-trough ratio is 2.4–11.7.
**The stationary model therefore needs a surge margin for planning,** or the time-varying analysis
named in the paper.

### 2. Failures come in outage episodes, not at a constant rate

| | Part 1 | Part 3 |
|---|---:|---:|
| Failure rate | 1.8 % | 7.3 % |
| P(fail \| previous request failed) | **62 %** | **27 %** |
| P(fail \| previous request succeeded) | 0.7 % | 5.7 % |
| Variance of the 5-min failure share / binomial variance | 62 | 69 |
| Failure share, busiest vs quietest rate decile | 0.32× | 11.3× |

Failures across *independent users' requests* are strongly clustered in time: consecutive requests
fail together 9–90 times more often than independence predicts. This points to service-side
outages rather than per-request bad luck. The dependence on load is real but **changes sign
between periods**. Spearman's ρ between window rate and failure share is −0.28 in part 1 and +0.39
in part 3. So neither a constant `p` nor a monotone `p(rho)` describes it. Two caveats: rate here is
this customer's request rate, not the provider's utilization, and the failures are HTTP-level
request failures, not agent action failures.

### 3. Measured latency supports the assumed token model

Linear fit `elapsed = a + b · prompt tokens + c · output tokens`, conversation mode, successful
requests:

| Model | Overhead `a` | Prefill `b` | Decode `c` | Median ms per output token | Assumed model's mean bias |
|---|---:|---:|---:|---:|---:|
| GPT-4 | 0.62 s | ≈ 0 ms/token | **31.1 ms/token** | 23.9 | −19 % |
| GPT-3.5 | 0.24 s | 0.10 ms/token | 12.7 ms/token | 12.9 | +89 % |
| *Assumed in all replays* | *0.25 s* | *0.2 ms/token* | *25 ms/token* | — | — |

* **The assumed coefficients match measured GPT-4 latency** (decode 25 vs 24–31 ms/token). AgentInstruct
  is GPT-4 data, so its replays used realistic service times. For the faster GPT-3.5 the assumed model
  is about 2× too slow. The decode-speed sensitivity run of Dataset 1 (12.5 and 50 ms/token) brackets
  both models, and its conclusions did not change.
* **Lognormal is the AIC-best family for measured response times** of both models (CV 1.35). This is
  the fourth independent dataset to confirm it.
* **Response time does not grow with this customer's load** (Spearman ρ ≈ −0.1 between window rate
  and measured/fitted ratio). From the client's side the provider behaves like a pool with ample
  capacity. Queueing at a self-hosted pool, which the model describes, is not observable in this
  trace.

### 4. Human chat sessions are heavier-tailed than geometric, unlike agent tasks

55,920 conversation sessions have 4.18 turns on average, but the variance is 50.8 against 13.3 for a
geometric law (3.8×). 35 % are single-turn, the p99 is 31 turns against 17 for geometric, and the
maximum is 436. The median think time between turns is 131 s, and prompts grow 8× over a session
(Spearman 0.51). Agent tasks in Datasets 1, 4 and 5 are *less* variable than geometric, while human
sessions are *more* variable. The geometric law sits between the two and fits neither.

## Dataset 8: Azure LLM inference trace 2024, one week (HPCA 2025)

```bash
python -m agentq run real_azure2024 --mode full --out results/real   # 3 min (+8 s one-off parse)
```

**Data.** We use the Azure LLM inference trace 2024 (Stojkovic et al., HPCA 2025 Best Paper,
*DynamoLLM*; CC-BY 4.0). It records every request to two production services, code completion
(**16.8M** requests, 27.8/s) and conversation (**27.3M**, 45.1/s), over **7 days** at microsecond
resolution, with context and generated tokens. It is 1.8 GB, checksummed and parsed once with pyarrow
into a `.npy` cache.

### Burstiness depends on aggregation scale (this corrects Dataset 6)

| Stream | Volume | 5-min windows above 1.25 × their hour's mean | Hours with at least one |
|---|---:|---:|---:|
| BurstGPT part 1 (one customer) | 0.27/s | 26 % | 96 % |
| BurstGPT part 3 (one customer) | 0.56/s | 21 % | 83 % |
| Azure 2023 conversation (1 hour) | 5.5/s | 18 % | — |
| **Azure 2024 code** | 27.8/s | **1.8 %** | **16 %** |
| **Azure 2024 conversation** | 45.1/s | **0.1 %** | **1.2 %** |

Dataset 6 concluded that within-hour surges are the norm. **That holds for low-volume streams such
as a single customer's traffic, not for high-volume aggregated services.** Within an hour, the
45 req/s conversation service is almost as smooth as Poisson at the 5-minute scale (98 % of hours
have no 1-minute window above 1.25 × the mean). The variation in these services sits at longer
scales: the diurnal peak-to-trough ratio is 9.0 (code) and 1.8 (conversation), and the busiest day
carries 3.5 × and 1.9 × the quietest. Those changes are handled by sizing each hour for its own rate.
Short-range burstiness is still far from Poisson (IDC 22 for code at 1 s; interarrival SCV 3.9 and
2.9). Lognormal is again the AIC-best service-time family.

### A validated rule for when the stationary model can be trusted

We replay the AgentInstruct `mixed` workload with real arrivals from 14 hours, chosen across the full
range of within-hour burstiness in both services, at `rho_L` = 0.7 and 0.8. Each hour is thinned at
random to 12,031 tasks, which keeps its rate profile, and we use 3 seeds. The predictor is
**`rho_L × peak ratio`**, where the peak ratio is the busiest minute's rate divided by the hour's mean.
A value of 1 or more means the pool is overloaded during the peak minute.

| `rho_L × peak ratio` | Points | Model mean error | Model p99 error |
|---|---:|---:|---:|
| < 0.95 | 17 | −6 % to +5 % | −10 % to +19 % |
| 0.95 – 1.0 | 5 | −20 % to −1 % | −36 % to +5 % |
| 1.0 – 1.05 | 2 | −29 % to −10 % | −53 % to −25 % |
| ≥ 1.05 | 4 | −79 % to −4 % | −90 % to −9 % |

Spearman's ρ between `rho_L × peak ratio` and the p99 error is **−0.90 (p = 4×10⁻¹¹, n = 28)**. The
share of surge windows is a weaker predictor (−0.57). **Rule: the stationary model can be trusted
when the target utilization times the hour's peak-minute ratio stays below about 0.95.** In that
regime the mean is within ±6 % and p99 within +19 % / −10 % across both services and both loads.
Above it, latency is underestimated by up to 90 %. This turns the arrival caveat of Datasets 1, 2 and 6
into a check that a planner can compute from their own trace before trusting the model.

## Model correction: the empirical step-count law

Every dataset with step counts (AgentInstruct, Who&When, SWE-Gym, MAST) rejects the geometric law.
`WorkflowConfig.step_pmf` replaces it with any law `P(N = n)`, such as an empirical pmf or a point
mass for a fixed pipeline. The success probability becomes `P_s = Σ πₙ sⁿ`, the number of started
steps `ν = Σ πₙ (1 − sⁿ)/(1 − s)`, and the latency mixture weights `wₙ ∝ πₙ sⁿ`. The simulator draws
`N` from the same law. Without `step_pmf` every code path is unchanged, so the paper results stay
bit-identical, and passing the geometric pmf explicitly reproduces the default model (unit-tested).
The `real_*_empirical` experiments repeat the validations with `step_law: empirical`.

```bash
python -m agentq run real_validation_empirical        --mode full --out results/real   # 9 min
python -m agentq run real_validation_swegym_empirical --mode full --out results/real   # 7 min
python -m agentq run real_azure2024_empirical         --mode full --out results/real   # 2 min
python -m agentq run real_mast                        --mode full --out results/real   # also reports the empirical law
```

### Before and after (p99 error, percent)

**Measured end-to-end latency, MAST (zero load).**

| Group | Geometric law p50 / p95 / p99 | Empirical law p50 / p95 / p99 |
|---|---:|---:|
| MetaGPT / GPT-4o | −24 / +79 / +116 | **+4 / −3 / −5** |
| MetaGPT / Claude | −23 / +97 / +156 | **+5 / −1 / 0** |
| MetaGPT / CodeLlama | −35 / +114 / +214 | −12 / +8 / +22 |
| MetaGPT / Qwen | −26 / +98 / +64 | −1 / +11 / −24 |
| ChatDev / CodeLlama | −32 / +117 / +193 | **−4 / +3 / +10** |
| ChatDev / GPT-4o | +107 / −20 / −3 | +184 / −50 / −41 |
| ChatDev / Qwen | −7 / +6 / +42 | +32 / −50 / −47 |

**Replay of real AgentInstruct trajectories, Poisson arrivals** (mean error was unchanged and is not shown):

| Environment | `rho_L` = 0.5: geometric → empirical | 0.8: geometric → empirical | 0.9: geometric → empirical |
|---|---:|---:|---:|
| OS | +84 → **+6** | +60 → **−9** | +20 → −33 |
| DB | +34 → **−7** | +33 → **−11** | +14 → −26 |
| ALFWorld | +53 → **−4** | +44 → **−10** | +15 → −29 |
| WebShop | +78 → **−2** | +68 → **−9** | +37 → −28 |
| KG | +145 → **−12** | +108 → −26 | +43 → −50 |
| Mixed | +23 → +26 | +12 → +15 | −11 → −8 |

**SWE-Gym replay, Poisson arrivals:** for all runs, p99 changes from +11 / +1 % to −22 / −30 % at
`rho_L` = 0.5 / 0.8, and for GPT-4o from +11 / +14 % to −21 / −20 %. **Azure 2024 planning rule:**
unchanged (Spearman −0.90; mean within ±6 % below 0.95).

### What the correction shows

1. **In every single-environment AgentInstruct workload, the p99 error at `rho_L ≤ 0.8` falls from
   +33 to +145 % to between −26 and +6 %**, against replays of real trajectories. On *measured* MetaGPT
   and ChatDev/CodeLlama latency it falls from +116 to +214 % to within ±5 % in 3 of 5 groups, and to
   within 24 % in the other two.
2. **At `rho_L = 0.9` the remaining underestimate (−26 to −50 %) is queueing-approximation error.** The
   model's own-assumption simulation under the empirical law shows the same size (−12 to −47 %), the
   pattern the paper reports for synthetic workloads near saturation. The geometric law's over-wide tail
   had hidden it.
3. **Correcting the step law uncovers the next error source: step times that depend on one another
   within a run.** In SWE-Gym (context grows 16× within a run), in two MAST ChatDev groups (run-mean
   variance 5.6× and 10.1× the i.i.d. value) and in the AgentInstruct `mixed` workload (environments with
   different service levels pooled into one), the empirical law *under*estimates p99 by 20–50 %, or
   leaves an overestimate in place (`mixed` +26 %). In these cases the geometric law had looked better
   only because its excess variance offset the missing correlation. A per-run or per-class service level,
   or step-dependent service times, is the next correction.
4. **Arrivals:** the Azure surge collapse is unaffected (−83 to −99 %), and the `rho_L × peak ratio`
   rule is unchanged. Arrival burstiness and the step law are separate error sources.

## Skipped: TaskBench (NeurIPS 2024)

TaskBench was evaluated and **not used**. Its workflow structures are synthetic: the dataset's
`data_engine.py` draws each sample's shape with fixed weights (`single: 3, chain: 7, dag: 8`) and fixed
weights for the number of tools. An LLM then writes a user request for the sampled tool graph
("Back-Instruct"), and humans verify the output. The mix of chains and DAGs is therefore a parameter
chosen by the authors, not an observation. Real fan-out evidence comes from BFCL-Live (Dataset 3) and
SWE-Gym (Dataset 5).

## Dataset 7: MAST multi-agent traces with measured step timings (NeurIPS 2025)

```bash
python -m agentq run real_mast --mode full --out results/real   # 1 min
```

**Data.** We use MAST/MAD (Cemri et al., NeurIPS 2025; CC-BY-4.0), `mcemri/MAST-Data` pinned to
revision `95118ac`: 200 MB holding 1,642 runs of 7 multi-agent frameworks on 8 benchmarks with 5 LLMs.
Each run carries binary annotations for the 14 MAST failure modes. The traces are raw framework logs,
and **three frameworks log wall-clock timestamps**:

* **ChatDev** (330 runs, 1 s resolution). LLM calls run one after another, and each is bracketed by
  the log line before its `HTTP Request: POST` line and that line itself. For GPT-4o
  (`api.openai.com`) the prompt and completion tokens are logged. CodeLlama and Qwen ran on a
  **self-hosted endpoint** (`localhost`) without token logging.
* **MetaGPT** (430 runs, 1 s). Each agent message is timestamped, so a step is the gap between
  messages.
* **OpenManus** (30 runs, 1 ms). The runs are **cut off at a 900 s limit**, so they are excluded from
  the latency test.

LLM steps account for **95–100 % of each run's wall-clock time** (median), so step latency is end-to-end
latency. Each run executed alone, so this is a **zero-load** test of the model's latency composition
(Eqs. 16–19 without queueing). It is the first test in this project against *measured end-to-end
latency* rather than a replay.

### Measured task latency vs the model (zero load)

Per framework and LLM, the model is calibrated on the runs themselves: a geometric step count with the
observed mean, and i.i.d. lognormal step times with the observed mean and CV. The control,
"empirical steps + i.i.d.", keeps the *observed* step-count distribution and draws step times i.i.d.
from the observed ones.

| Framework / LLM | Runs | Steps: mean (variance) | Step time: mean, CV | Model p50 / p95 / p99 error | Empirical steps + i.i.d. p50 / p95 / p99 error |
|---|---:|---:|---:|---:|---:|
| MetaGPT / GPT-4o | 130 | 5.0 (0.2) | 11.5 s, 0.67 | −24 / +79 / +116 % | +4 / −2 / −4 % |
| MetaGPT / Claude | 100 | 5.0 (0) | 8.0 s, 0.54 | −23 / +97 / +156 % | +5 / −10 / +30 % |
| MetaGPT / Qwen | 100 | 5.0 (0) | 18.1 s, 0.77 | −26 / +98 / +64 % | 0 / +4 / −10 % |
| MetaGPT / CodeLlama | 100 | 5.0 (0) | 35.8 s, 0.53 | −35 / +114 / +214 % | −9 / +4 / +6 % |
| ChatDev / CodeLlama | 100 | 54.2 (7.8) | 15.1 s, 1.77 | −32 / +117 / +193 % | −2 / +2 / +2 % |
| ChatDev / GPT-4o | 130 | 13.8 (1.7) | 16.2 s, 2.28 | +107 / −20 / −3 % | +199 / −46 / −47 % |
| ChatDev / Qwen | 100 | 52.8 (3.1) | 16.6 s, 1.76 | −7 / +6 / +42 % | +32 / −49 / −48 % |

The model's mean is exact by construction (`n̄ · E[S]`) and is not shown.

1. **Real multi-agent frameworks run fixed-length pipelines.** MetaGPT always takes 5 steps, and
   ChatDev takes 13.8 ± 1.3 steps (GPT-4o) or about 53 (open models). The geometric law has 10–370×
   the real step-count variance, so **the model overestimates p95/p99 by 64–214 % and underestimates
   the median by 23–35 %** in 5 of 7 groups.
2. **With the real step-count law, i.i.d. step times reproduce measured latency.** In those 5 groups,
   the empirical-steps + i.i.d. control is within ±10 % of measured p50, p95 and p99. The one exception
   is MetaGPT/Claude p99 (+30 %, from 100 runs). Measured step times are again lognormal (AIC-best in
   5 of 7 groups, gamma in the others). **The model's composition machinery is correct on measured
   data once the step law is right.**
3. **The exception is step times that are correlated within a run** (ChatDev with GPT-4o and Qwen).
   The variance of a run's mean step time is 5.6× (GPT-4o) and 10.1× (Qwen) what independent steps
   would give, so slow runs are slow throughout. The i.i.d. control then underestimates p95/p99 by
   46–49 %. The model's overdispersed geometric law happens to compensate (p99 −3 % and +42 %): errors
   of opposite sign cancel again, as in Datasets 2 and 5. A run-level random effect (a per-run speed
   factor) would capture this.
4. **Measured GPT-4o speed:** the median is 17 ms per completion token (ChatDev calls with 50+ tokens),
   faster than BurstGPT's GPT-4 (24–31 ms). The least-squares fit is unstable on these heavy-tailed,
   correlated call times (negative intercept, R² 0.48), so we report the median only. 3 of 1,798
   GPT-4o calls returned HTTP 5xx.

### Failure modes (all 1,642 runs)

75 % of runs show at least one MAST failure mode (AG2 48 %; ChatDev, MetaGPT, OpenManus, AppWorld
and HyperAgent 90–97 %). The most common are **step repetition (1.3, 41 %)**, incorrect verification
(3.3, 40 %), reasoning–action mismatch (2.6, 37 %), no or incomplete verification (3.2, 37 %) and
disobeying the task specification (1.1, 36 %). The modes split between ones a runtime can detect
(step repetition, which matches the SWE-Gym stuck-loop aborts) and ones that produce **silent
failures** (verification modes 3.2/3.3, which match the SWE-Gym wrong-patch finishes). Only 30 AG2
traces record whether the answer was correct, so failure modes cannot be linked to outcomes or cost.

### Next steps

* Replace the geometric law with a step-count law fitted to the data (for example a shifted
  negative binomial or the empirical pmf). The convolution algorithm supports this with little change.
* Model failure clustering in the retry mixture (a two-state failure chain).
* Compare the Azure-arrival replay with the paper's quasi-static burst bracket, using a
  time-varying rate fitted to the trace.
* Add an outcome-conditioned cost term: calls spent on silent failures, which make up 78 % of
  SWE-Gym's calls, and a step law fitted to successful runs only.
* All planned datasets are done. An instrumented agent run under load would test queueing at a shared
  pool, which no public dataset does (MAST measures latency only at zero load).
* Model changes indicated by the data: an empirical or fixed-length step law, a per-run speed factor
  for correlated step times, and a cost term for silent failures.
