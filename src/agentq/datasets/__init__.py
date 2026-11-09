"""Loaders for real, public datasets used to calibrate and test the model.

Unlike everything else in the package, these modules read *measured* data:

* :mod:`agentq.datasets.agentinstruct` - 1,866 GPT-4 agent trajectories on AgentBench tasks
  (Zeng et al., 2023, "AgentTuning"), giving step counts, tool calls, failed actions and
  per-call token counts.
* :mod:`agentq.datasets.azure` - the Azure LLM inference trace 2023 (conversation service;
  Patel et al., 2024, "Splitwise"), giving one hour of real request arrival timestamps.

See ``docs/real_data.md`` for what each dataset can and cannot identify.
"""

from agentq.datasets.agentinstruct import (
    ENVIRONMENTS,
    AttemptRecord,
    Trajectory,
    load_agentinstruct,
    parse_conversation,
    to_attempt_rows,
)
from agentq.datasets.azure import arrival_statistics, load_azure_arrivals, rate_surges
from agentq.datasets.bfcl import empirical_pmf, fanout_counts
from agentq.datasets.fetch import DATASETS, fetch_datasets
from agentq.datasets.mooncake import load_mooncake, prefix_linkage, timestamp_resolution
from agentq.datasets.service import TokenLatencyModel
from agentq.datasets.whowhen import Run, load_whowhen

__all__ = [
    "DATASETS",
    "ENVIRONMENTS",
    "AttemptRecord",
    "Run",
    "TokenLatencyModel",
    "Trajectory",
    "arrival_statistics",
    "empirical_pmf",
    "fanout_counts",
    "fetch_datasets",
    "load_agentinstruct",
    "load_azure_arrivals",
    "load_mooncake",
    "load_whowhen",
    "parse_conversation",
    "prefix_linkage",
    "rate_surges",
    "timestamp_resolution",
    "to_attempt_rows",
]
