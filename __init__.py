"""agentq: queueing and branching-process performance models for multi-agent LLM workflows.

Core entry points:

* :func:`agentq.model.predict` - latency distribution, success probability and cost of a chain
* :func:`agentq.trees.predict_tree` - sub-agent trees (fixed point)
* :func:`agentq.bursts.bracket` - Poisson / quasi-static bracket for MMPP arrivals
* :func:`agentq.planning.max_rate_for_p99` - SLO-driven capacity planning
* :func:`agentq.sim.simulate_chain` / :func:`agentq.sim.simulate_tree` - reference simulators
"""

from agentq.config import PoolConfig, SystemConfig, WorkflowConfig, load_profiles
from agentq.model import ModelOptions, Prediction, predict
from agentq.structure import pool_loads, rate_for_llm_utilization, stability_limit, structural_quantities

__version__ = "1.0.0"

__all__ = [
    "ModelOptions",
    "PoolConfig",
    "Prediction",
    "SystemConfig",
    "WorkflowConfig",
    "__version__",
    "load_profiles",
    "pool_loads",
    "predict",
    "rate_for_llm_utilization",
    "stability_limit",
    "structural_quantities",
]
