"""Discrete-event simulator for sub-agent trees (paper Section 7, "Sub-Agent Trees").

Every agent runs a chain; a successful agent delegates to ``D ~ Poisson(m)`` sub-agents and
completes when all of them complete. A tree fails as soon as any agent exhausts its retries.

Behavior inherited from the paper's implementation (documented in docs/assumptions.md, A11):
agents of an already-failed tree keep running and consuming capacity, and a failed agent's
parent is never released. This affects load, not the latency of successful trees.
"""

from __future__ import annotations

import heapq
from dataclasses import asdict, dataclass

import numpy as np

from agentq.config import SystemConfig
from agentq.sim.arrivals import poisson_arrivals
from agentq.sim.chain import Station


@dataclass
class TreeSimResult:
    """Statistics of one tree simulation (trees arriving after warm-up)."""

    mean: float
    p50: float
    p95: float
    p99: float
    tree_success: float
    agents_per_tree: float
    n_trees: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def simulate_tree(
    system: SystemConfig,
    tree_rate: float,
    delegation_mean: float,
    n_trees: int = 20_000,
    seed: int = 0,
    warmup_fraction: float = 0.1,
) -> TreeSimResult:
    """Simulate ``n_trees`` Poisson-arriving trees with Poisson(``delegation_mean``) delegation."""
    if tree_rate <= 0 or n_trees < 10 or delegation_mean < 0:
        raise ValueError("invalid tree simulation parameters")
    rng = np.random.default_rng(seed)
    horizon = n_trees / tree_rate
    arr = poisson_arrivals(rng, tree_rate, n_trees)
    llm, tool = Station(system.llm, rng), Station(system.tool, rng)
    wf = system.workflow
    p, r_max, q, m = wf.failure_prob, wf.max_retries, wf.q, delegation_mean
    poisson_fan = wf.fanout_kind == "poisson"

    events: list[tuple[float, int, str, int]] = []
    seq = 0

    def push(t: float, kind: str, job: int) -> None:
        nonlocal seq
        seq += 1
        heapq.heappush(events, (t, seq, kind, job))

    for i, t in enumerate(arr):
        push(float(t), "arr", i)
    # agent id -> [tree, parent, steps_left, attempt, pending_tools, pending_children]
    agents: dict[int, list] = {}
    trees: dict[int, list] = {}  # tree id -> [t0, failed]
    results: list[tuple[float, float, int]] = []
    n_agents = 0
    next_id = 0

    def submit(st: Station, now: float, job: int, kind: str) -> None:
        if st.busy < st.k:
            st.busy += 1
            d = st.service()
            push(now + d, kind, job)
            st.busy_time += d
        else:
            st.queue.append((job, kind))

    def release(st: Station, now: float) -> None:
        st.busy -= 1
        if st.queue:
            job, kind = st.queue.popleft()
            st.busy += 1
            d = st.service()
            push(now + d, kind, job)
            st.busy_time += d

    def new_agent(now: float, tree: int, parent: int | None) -> None:
        nonlocal n_agents, next_id
        next_id += 1
        n_agents += 1
        agents[next_id] = [tree, parent, int(rng.geometric(q)), 0, 0, 0]
        submit(llm, now, next_id, "llm_done")

    def agent_done(now: float, aid: int) -> None:
        tree, parent = agents[aid][0], agents[aid][1]
        del agents[aid]
        if parent is None:
            tr = trees[tree]
            if not tr[1]:
                results.append((tr[0], now - tr[0], 1))
        else:
            pa = agents[parent]
            pa[5] -= 1
            if pa[5] == 0:
                agent_done(now, parent)

    def finish_attempt(now: float, aid: int) -> None:
        ag = agents[aid]
        if rng.random() < p:
            if ag[3] < r_max:
                ag[3] += 1
                submit(llm, now, aid, "llm_done")
                return
            tr = trees[ag[0]]
            if not tr[1]:
                tr[1] = True
                results.append((tr[0], now - tr[0], 0))
            return
        ag[2] -= 1
        ag[3] = 0
        if ag[2] > 0:
            submit(llm, now, aid, "llm_done")
            return
        d = int(rng.poisson(m))
        if d == 0:
            agent_done(now, aid)
        else:
            ag[5] = d
            for _ in range(d):
                new_agent(now, ag[0], aid)

    while events:
        now, _, kind, job = heapq.heappop(events)
        if kind == "arr":
            trees[job] = [now, False]
            new_agent(now, job, None)
        elif kind == "llm_done":
            release(llm, now)
            ag = agents[job]
            k = int(rng.poisson(wf.fanout)) if poisson_fan else int(wf.fanout)
            if k == 0:
                finish_attempt(now, job)
            else:
                ag[4] = k
                for _ in range(k):
                    submit(tool, now, job, "tool_done")
        else:
            release(tool, now)
            ag = agents[job]
            ag[4] -= 1
            if ag[4] == 0:
                finish_attempt(now, job)

    res = np.array(results)
    res = res[res[:, 0] >= warmup_fraction * horizon]
    ok = res[res[:, 2] == 1, 1]
    if ok.size == 0:
        raise RuntimeError("no successful trees after warm-up; increase n_trees")
    return TreeSimResult(
        mean=float(ok.mean()),
        p50=float(np.percentile(ok, 50)),
        p95=float(np.percentile(ok, 95)),
        p99=float(np.percentile(ok, 99)),
        tree_success=float(res[:, 2].mean()),
        agents_per_tree=n_agents / n_trees,
        n_trees=int(len(res)),
    )
