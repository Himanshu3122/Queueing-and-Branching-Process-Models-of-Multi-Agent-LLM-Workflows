"""Paper experiments (Tables 3-12, Figures 4-8) and their runner."""

from agentq.experiments.paper import EXPERIMENTS
from agentq.experiments.runner import experiment_params, run_all, run_experiment

__all__ = ["EXPERIMENTS", "experiment_params", "run_all", "run_experiment"]
