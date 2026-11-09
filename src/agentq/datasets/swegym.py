"""SWE-Gym sampled OpenHands trajectories (Pan et al., ICML 2025), with both outcomes.

Dataset ``SWE-Gym/OpenHands-Sampled-Trajectories`` on the Hugging Face Hub, pinned to revision
``baf3a4e`` (3 parquet shards, 301 MB, 6,055 trajectories). The card states no licence; the
companion SFT set is MIT. Each row is one OpenHands CodeAct run on a real GitHub issue, with
``run_id`` (model, iteration cap ``maxiter``, sampling temperature), ``resolved`` (hidden tests
pass) and the full message list with structured ``tool_calls``.

Mapping to the model (one assistant message = one LLM call = one *attempt*):

* **tool calls** ``K``: the attempt's tool calls, excluding the ``finish`` pseudo-tool;
* **failed attempt** (strict): a tool observation of the attempt is a tool-usage error - editor
  ``ERROR:``, or a bash exit code of 2, 126 or 127. *Broad* additionally counts any non-zero exit,
  which includes the reproduction scripts agents run on purpose to watch the bug fail;
* **tokens**: context = all earlier messages (tool schemas excluded), output = the message plus
  its tool-call arguments, counted with GPT-4o's ``o200k_base`` encoding for every model.

How a run ends: ``finish`` (the agent declares completion), ``cap`` (hit ``maxiter``), or
``abort`` (stopped earlier without ``finish``, typically by OpenHands' stuck-loop detector after
repeated identical actions). ``empty_patch`` means no code change was produced.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from agentq.datasets.agentinstruct import AttemptRecord, Trajectory

log = logging.getLogger(__name__)
_RUN = re.compile(r"^(?P<model>.+?)_maxiter_(?P<cap>\d+)_.*?(?:-t(?P<temp>\d+))?-run_\d+$")
_EXIT = re.compile(r"exit code (\d+)")
STRICT_EXIT = {2, 126, 127}


def parse_run_id(run_id: str) -> dict[str, Any]:
    m = _RUN.match(run_id)
    if not m:
        return {"model": run_id, "maxiter": None, "temperature": None}
    temp = m.group("temp")
    return {
        "model": m.group("model"),
        "maxiter": int(m.group("cap")),
        "temperature": (int(temp) / 10 if temp else None),
    }


def _observation_failure(text: str) -> tuple[bool, bool]:
    """(strict, broad) failure flags of one tool observation."""
    body = text[len("OBSERVATION:") :].lstrip() if text.startswith("OBSERVATION:") else text
    strict = body.startswith("ERROR:")
    codes = [int(c) for c in _EXIT.findall(text)]
    last = codes[-1] if codes else 0
    strict = strict or last in STRICT_EXIT
    return strict, strict or last != 0


def _tail_repeats(actions: list[str]) -> int:
    """Length of the run of identical consecutive actions at the end of a trajectory."""
    if not actions:
        return 0
    n = 1
    for a, b in zip(actions[-2::-1], actions[:0:-1], strict=False):
        if a != b:
            break
        n += 1
    return n


def extract_record(row: dict[str, Any], count_tokens: Callable[[str], int]) -> dict[str, Any]:
    """Compact per-run record (see module docstring)."""
    msgs = row["messages"]
    toks = []
    for m in msgs:
        text = m.get("content") or ""
        for c in m.get("tool_calls") or []:
            text += c["function"]["name"] + (c["function"].get("arguments") or "")
        toks.append(count_tokens(text))
    attempts, actions = [], []
    ctx = 0
    i = 0
    finished = False
    while i < len(msgs):
        m = msgs[i]
        if m["role"] != "assistant":
            ctx += toks[i]
            i += 1
            continue
        calls = m.get("tool_calls") or []
        names = [c["function"]["name"] for c in calls]
        finished = finished or "finish" in names
        actions.append(json.dumps([[c["function"]["name"], c["function"].get("arguments")] for c in calls]))
        j = i + 1
        strict = broad = False
        while j < len(msgs) and msgs[j]["role"] != "assistant":
            if msgs[j]["role"] == "tool":
                s, b = _observation_failure(msgs[j].get("content") or "")
                strict, broad = strict or s, broad or b
            j += 1
        attempts.append([ctx, toks[i], sum(1 for n in names if n != "finish"), int(strict), int(broad)])
        ctx += toks[i]
        i += 1
    meta = parse_run_id(row["run_id"])
    report = (row.get("test_result") or {}).get("report") or {}
    cap = meta["maxiter"]
    ending = "finish" if finished else ("cap" if cap is not None and len(attempts) >= cap else "abort")
    return {
        "run_id": row["run_id"],
        **meta,
        "instance_id": row["instance_id"],
        "resolved": bool(row["resolved"]),
        "empty_patch": bool(report.get("empty_generation", False)),
        "ending": ending,
        "tail_repeats": _tail_repeats(actions),
        "attempts": attempts,
    }


def extract_swegym(data_dir: str | Path, cache: str | Path, count_tokens: Callable[[str], int] | None = None) -> Path:
    """Stream the parquet shards once and cache compact records as gzipped JSON lines."""
    import pyarrow.parquet as pq

    cache = Path(cache)
    if cache.is_file():
        return cache
    if count_tokens is None:
        import tiktoken

        enc = tiktoken.get_encoding("o200k_base")

        def count_tokens(text: str) -> int:
            return len(enc.encode(text, disallowed_special=()))

    shards = sorted(Path(data_dir).glob("train.raw-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no SWE-Gym shards in {data_dir}; run `python scripts/fetch_datasets.py` first")
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".part")
    n = 0
    with gzip.open(tmp, "wt", encoding="utf-8") as out:
        for shard in shards:
            pf = pq.ParquetFile(shard)
            cols = ["instance_id", "run_id", "resolved", "messages", "test_result"]
            for batch in pf.iter_batches(batch_size=64, columns=cols):
                for row in batch.to_pylist():
                    out.write(json.dumps(extract_record(row, count_tokens)) + "\n")
                    n += 1
            log.info("extracted %s (%d runs so far)", shard.name, n)
    tmp.replace(cache)
    return cache


def load_records(cache: str | Path) -> list[dict[str, Any]]:
    with gzip.open(cache, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def to_trajectory(rec: dict[str, Any]) -> Trajectory:
    """Strict failures become retries of the same step."""
    out, step, attempt = [], 1, 0
    for ctx, gen, k, strict, _broad in rec["attempts"]:
        out.append(AttemptRecord(step, attempt, bool(strict), int(k), int(ctx), int(gen)))
        step, attempt = (step, attempt + 1) if strict else (step + 1, 0)
    return Trajectory(env=rec["model"], traj_id=rec["run_id"], attempts=tuple(out), success=rec["resolved"])


def swegym_workloads(cache: str | Path, names: Iterable[str]) -> dict[str, list[Trajectory]]:
    """Workloads ``all`` and per model (``gpt-4o-...``, ``claude-...``); runs without attempts dropped."""
    recs = [r for r in load_records(cache) if r["attempts"]]
    out: dict[str, list[Trajectory]] = {}
    for name in names:
        sel = recs if name == "all" else [r for r in recs if r["model"].startswith(name)]
        out[name] = [to_trajectory(r) for r in sel]
    return out
