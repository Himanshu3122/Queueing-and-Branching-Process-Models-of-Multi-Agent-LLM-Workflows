"""Trace-driven simulator: replay recorded agent trajectories through the shared pools.

Same pool mechanics as :mod:`agentq.sim.chain` (FIFO, ``k`` servers per pool, fork-join of the
tool calls of an attempt), but the workflow is not drawn from the model's distributions. Each
arriving task replays one recorded trajectory exactly: its number of attempts, which attempts
failed, how many tool calls each issued, and the LLM service time of each call (from its real
token counts). Only tool service times are drawn (lognormal from the tool pool), because the
datasets do not record them.

Arrival times are supplied by the caller (Poisson or a rescaled real trace).
"""

from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Sequence

import numpy as np

from agentq.config import PoolConfig
from agentq.queueing import lognormal_params
from agentq.sim.chain import SimResult

# One recorded task: per attempt (llm_service_seconds, n_tool_calls); plus the task outcome.
ReplayTask = tuple[tuple[tuple[float, int], ...], bool]


def simulate_replay(
    tasks: Sequence[ReplayTask],
    arrival_times: np.ndarray,
    llm_servers: int,
    tool: PoolConfig,
    seed: int = 0,
    warmup_fraction: float = 0.1,
) -> SimResult:
    """Replay ``tasks[i]`` arriving at ``arrival_times[i]``.

    Args:
        tasks: One recorded trajectory per arrival (same length as ``arrival_times``).
        arrival_times: Non-decreasing arrival times in seconds.
        llm_servers: ``k_L``; LLM service times come from the trajectories.
        tool: Tool pool (servers, lognormal service mean and CV).
        seed: Seed for the tool service times.
        warmup_fraction: Tasks arriving before this fraction of the horizon are discarded.

    Returns:
        A :class:`SimResult` (latency of successful tasks, success fraction, utilizations).
    """
    arr = np.asarray(arrival_times, dtype=float)
    if len(tasks) != arr.size or arr.size < 10:
        raise ValueError("need one task per arrival and at least 10 arrivals")
    if llm_servers < 1 or not 0 <= warmup_fraction < 1:
        raise ValueError("invalid simulation parameters")
    rng = np.random.default_rng(seed)
    mu_t, sg_t = lognormal_params(tool.mean_service, tool.cv)

    events: list[tuple[float, int, int, int]] = []  # (time, seq, kind, task); kind 0=arr 1=llm 2=tool
    seq = 0

    def push(t: float, kind: int, job: int) -> None:
        nonlocal seq
        seq += 1
        heapq.heappush(events, (t, seq, kind, job))

    for i, t in enumerate(arr):
        push(float(t), 0, i)

    busy = {1: 0, 2: 0}
    servers = {1: llm_servers, 2: tool.servers}
    queues: dict[int, deque[int]] = {1: deque(), 2: deque()}
    busy_time = {1: 0.0, 2: 0.0}
    cursor = np.zeros(arr.size, dtype=np.int64)  # next attempt index per task
    pending = np.zeros(arr.size, dtype=np.int64)  # outstanding tool calls per task
    results: list[tuple[float, float, int]] = []
    calls = 0

    def duration(pool: int, job: int) -> float:
        if pool == 1:
            return tasks[job][0][int(cursor[job])][0]
        return float(rng.lognormal(mu_t, sg_t))

    def submit(pool: int, now: float, job: int) -> None:
        if busy[pool] < servers[pool]:
            busy[pool] += 1
            d = duration(pool, job)
            busy_time[pool] += d
            push(now + d, pool, job)
        else:
            queues[pool].append(job)

    def release(pool: int, now: float) -> None:
        busy[pool] -= 1
        if queues[pool]:
            job = queues[pool].popleft()
            busy[pool] += 1
            d = duration(pool, job)
            busy_time[pool] += d
            push(now + d, pool, job)

    def next_attempt(now: float, job: int) -> None:
        nonlocal calls
        if cursor[job] >= len(tasks[job][0]):
            results.append((float(arr[job]), now - float(arr[job]), int(tasks[job][1])))
            return
        calls += 1
        submit(1, now, job)

    while events:
        now, _, kind, job = heapq.heappop(events)
        if kind == 0:
            next_attempt(now, job)
        elif kind == 1:
            release(1, now)
            k = tasks[job][0][int(cursor[job])][1]
            if k == 0:
                cursor[job] += 1
                next_attempt(now, job)
            else:
                pending[job] = k
                for _ in range(k):
                    submit(2, now, job)
        else:
            release(2, now)
            pending[job] -= 1
            if pending[job] == 0:
                cursor[job] += 1
                next_attempt(now, job)

    res = np.array(results)
    horizon = float(arr[-1])
    res = res[res[:, 0] >= arr[0] + warmup_fraction * (horizon - arr[0])]
    ok = res[res[:, 2] == 1, 1]
    if ok.size == 0:
        raise RuntimeError("no successful tasks after warm-up")
    end = max(horizon, 1e-9)
    return SimResult(
        mean=float(ok.mean()),
        p50=float(np.percentile(ok, 50)),
        p95=float(np.percentile(ok, 95)),
        p99=float(np.percentile(ok, 99)),
        success_prob=float(res[:, 2].mean()),
        llm_calls_per_task=calls / arr.size,
        n_tasks=int(len(res)),
        rho_llm=busy_time[1] / (llm_servers * end),
        rho_tool=busy_time[2] / (tool.servers * end),
    )
