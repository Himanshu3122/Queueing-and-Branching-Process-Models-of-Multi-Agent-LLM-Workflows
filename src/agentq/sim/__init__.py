"""Discrete-event reference simulators used to validate the analytical model."""

from agentq.sim.arrivals import mmpp_arrivals, poisson_arrivals
from agentq.sim.chain import SimResult, simulate_chain
from agentq.sim.tree import TreeSimResult, simulate_tree

__all__ = ["SimResult", "TreeSimResult", "mmpp_arrivals", "poisson_arrivals", "simulate_chain", "simulate_tree"]
