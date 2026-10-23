"""Discrete-event reference simulator for agent chains (paper Section 6, "Reference simulator").

Implements the workflow of Section 3.1 exactly: FIFO queues with ``k`` servers per pool,
lognormal service times, geometric step counts, fixed or Poisson tool fan-out with a join, and
independent per-attempt failures with at most ``R`` retries. The random-number draw order is
identical to the implementation used for the paper, so the same seeds give the same results.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from agentq.bursts import MMPP2
from agentq.config import PoolConfig, SystemConfig
from agentq.queueing import lognormal_params
from agentq.sim.arrivals import mmpp_arrivals, poisson_arrivals


class Station:
    """A FIFO pool of ``k`` identical servers with lognormal service."""

    def __init__(self, pool: PoolConfig, rng: np.random.Generator) -> None:
        self.k = pool.servers
        self.busy = 0
        self.queue: deque[tuple[int, str]] = deque()
        self.mu, self.sigma = lognormal_params(pool.mean_service, pool.cv)
        self.rng = rng
        self.busy_time = 0.0

    def service(self) -> float:
        return float(self.rng.lognormal(self.mu, self.sigma))


@dataclass
class SimResult:
    """Statistics of one simulation run (tasks arriving after the warm-up period)."""

    mean: float
    p50: float
    p95: float
    p99: float
    success_prob: float
    llm_calls_per_task: float
    n_tasks: int
    rho_llm: float
    rho_tool: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def simulate_chain(
    system: SystemConfig,
    task_rate: float,
    n_tasks: int = 30_000,
    seed: int = 0,
    arrivals: MMPP2 | None = None,
    warmup_fraction: float = 0.1,
) -> SimResult:
    """Simulate ``n_tasks`` tasks (or an MMPP stream over the same horizon) and summarize.

    Args:
        system: Workflow and pool configuration.
        task_rate: Mean task arrival rate.
        n_tasks: Number of Poisson arrivals; for MMPP the horizon is ``n_tasks / task_rate``.
        seed: Seed for ``numpy.random.default_rng``.
        arrivals: Optional MMPP arrival process (its ``rate`` should equal ``task_rate``).
        warmup_fraction: Tasks arriving before this fraction of the horizon are discarded.

    Returns:
        Latency statistics of successful tasks, success probability, LLM calls per task and
        pool utilizations (busy time / (servers x last arrival time), as in the paper's code).
    """
    if task_rate <= 0 or n_tasks < 10 or not 0 <= warmup_fraction < 1:
        raise ValueError("invalid simulation parameters")
    rng = np.random.default_rng(seed)
    horizon = n_tasks / task_rate
    arr = mmpp_arrivals(rng, arrivals, horizon) if arrivals is not None else poisson_arrivals(rng, task_rate, n_tasks)
    if arr.size == 0:
        raise ValueError("no arrivals generated")
    llm, tool = Station(system.llm, rng), Station(system.tool, rng)
    wf = system.workflow
    p, r_max, q = wf.failure_prob, wf.max_retries, wf.q
    poisson_fan = wf.fanout_kind == "poisson"

    events: list[tuple[float, int, str, int]] = []
    seq = 0

    def push(t: float, kind: str, job: int) -> None:
        nonlocal seq
        seq += 1
        heapq.heappush(events, (t, seq, kind, job))

    for i, t in enumerate(arr):
        push(float(t), "arr", i)
    tasks: dict[int, list[float]] = {}  # id -> [t0, steps_left, attempt, pending_tools]
    results: list[tuple[float, float, int]] = []
    calls = 0

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

    def start_attempt(now: float, tid: int) -> None:
        nonlocal calls
        calls += 1
        submit(llm, now, tid, "llm_done")

    def finish_attempt(now: float, tid: int) -> None:
        tk = tasks[tid]
        if rng.random() < p:
            if tk[2] < r_max:
                tk[2] += 1
                start_attempt(now, tid)
                return
            results.append((tk[0], now - tk[0], 0))
            del tasks[tid]
            return
        tk[1] -= 1
        tk[2] = 0
        if tk[1] == 0:
            results.append((tk[0], now - tk[0], 1))
            del tasks[tid]
        else:
            start_attempt(now, tid)

    while events:
        now, _, kind, job = heapq.heappop(events)
        if kind == "arr":
            tasks[job] = [now, float(rng.geometric(q)), 0, 0]
            start_attempt(now, job)
        elif kind == "llm_done":
            release(llm, now)
            tk = tasks[job]
            k = int(rng.poisson(wf.fanout)) if poisson_fan else int(wf.fanout)
            if k == 0:
                finish_attempt(now, job)
            else:
                tk[3] = k
                for _ in range(k):
                    submit(tool, now, job, "tool_done")
        else:
            release(tool, now)
            tk = tasks[job]
            tk[3] -= 1
            if tk[3] == 0:
                finish_attempt(now, job)

    res = np.array(results)
    res = res[res[:, 0] >= warmup_fraction * horizon]
    ok = res[res[:, 2] == 1, 1]
    if ok.size == 0:
        raise RuntimeError("no successful tasks after warm-up; increase n_tasks")
    end = max(float(arr[-1]), 1e-9)
    return SimResult(
        mean=float(ok.mean()),
        p50=float(np.percentile(ok, 50)),
        p95=float(np.percentile(ok, 95)),
        p99=float(np.percentile(ok, 99)),
        success_prob=float(res[:, 2].mean()),
        llm_calls_per_task=calls / len(arr),
        n_tasks=int(len(res)),
        rho_llm=llm.busy_time / (system.llm.servers * end),
        rho_tool=tool.busy_time / (system.tool.servers * end),
    )
