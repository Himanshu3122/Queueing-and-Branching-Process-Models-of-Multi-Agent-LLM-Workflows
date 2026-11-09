"""MAST / MAD traces (Cemri et al., NeurIPS 2025): real multi-agent runs, failure-mode annotations,
and - for three frameworks - wall-clock timestamps.

``mcemri/MAST-Data`` on the Hugging Face Hub (CC-BY-4.0), ``MAD_full_dataset.json`` pinned to revision
``95118ac``: 1,642 traces from 7 frameworks (AG2, MetaGPT, ChatDev, Magentic, AppWorld, HyperAgent,
OpenManus), 8 benchmarks and 5 LLMs, each with binary annotations for the 14 MAST failure modes.

The ``trajectory`` is the framework's raw log. Three frameworks log timestamps that this module
parses into per-step durations:

* **ChatDev** - sequential LLM calls; a call starts at the log line before its
  ``HTTP Request: POST ...`` line and ends there (1 s resolution). The following
  ``OpenAI_Usage_Info`` block gives its prompt and completion tokens; non-200 responses are kept.
* **OpenManus** - millisecond log lines; a call ends at ``Token usage: Input=..., Completion=...`` and
  starts at the previous log line.
* **MetaGPT** - one timestamp per agent message (1 s); a step is the gap between consecutive
  messages, so it includes any non-LLM work of that step.

AG2 traces have no timestamps but record whether the final answer was ``correct``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

MODES = ("1.1", "1.2", "1.3", "1.4", "1.5", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "3.1", "3.2", "3.3")
CATEGORIES = {"specification": MODES[:5], "inter_agent": MODES[5:11], "verification": MODES[11:]}

_CD_LINE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) INFO\] (.*)$", re.M)
_CD_HTTP = re.compile(r'HTTP Request: POST \S+ "HTTP/1\.1 (\d{3})')
_CD_TOK = re.compile(r"prompt_tokens: (\d+)\ncompletion_tokens: (\d+)")
_OM_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \| \w+\s*\| (.*)$", re.M)
_OM_TOK = re.compile(r"Token usage: Input=(\d+), Completion=(\d+)")
_MG_LINE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (FROM: |NEW MESSAGES)", re.M)


@dataclass
class TimedTrace:
    """Measured timings of one run."""

    mas: str
    llm: str
    benchmark: str
    total_s: float
    step_s: list[float]
    tokens: list[tuple[int, int]] = field(default_factory=list)  # (prompt, completion) per LLM call
    http_status: dict[str, int] = field(default_factory=dict)


def load_mast(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `python scripts/fetch_datasets.py` first")
    return json.loads(path.read_text(encoding="utf-8"))


def _chatdev(rec: dict[str, Any]) -> TimedTrace | None:
    text = rec["trace"]["trajectory"]
    lines = [
        (datetime.strptime(m.group(1), "%Y-%d-%m %H:%M:%S"), m.group(2), m.start()) for m in _CD_LINE.finditer(text)
    ]
    if len(lines) < 2:
        return None
    steps, tokens, status = [], [], {}
    for i in range(1, len(lines)):
        hit = _CD_HTTP.search(lines[i][1])
        if not hit:
            continue
        code = hit.group(1)
        status[code] = status.get(code, 0) + 1
        if code != "200":
            continue
        steps.append((lines[i][0] - lines[i - 1][0]).total_seconds())
        # the token counts follow the "[ts] INFO] **[OpenAI_Usage_Info Receive]**" header line (lines[i + 1])
        tok = _CD_TOK.search(text, lines[i][2], lines[i + 2][2] if i + 2 < len(lines) else len(text))
        tokens.append((int(tok.group(1)), int(tok.group(2))) if tok else (-1, -1))
    total = (lines[-1][0] - lines[0][0]).total_seconds()
    return TimedTrace(rec["mas_name"], rec["llm_name"], rec["benchmark_name"], total, steps, tokens, status)


def _openmanus(rec: dict[str, Any]) -> TimedTrace | None:
    text = rec["trace"]["trajectory"]
    lines = [(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f"), m.group(2)) for m in _OM_LINE.finditer(text)]
    if len(lines) < 2:
        return None
    steps, tokens = [], []
    for i in range(1, len(lines)):
        hit = _OM_TOK.search(lines[i][1])
        if hit:
            steps.append((lines[i][0] - lines[i - 1][0]).total_seconds())
            tokens.append((int(hit.group(1)), int(hit.group(2))))
    total = (lines[-1][0] - lines[0][0]).total_seconds()
    return TimedTrace(rec["mas_name"], rec["llm_name"], rec["benchmark_name"], total, steps, tokens)


def _metagpt(rec: dict[str, Any]) -> TimedTrace | None:
    stamps = [datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") for m in _MG_LINE.finditer(rec["trace"]["trajectory"])]
    if len(stamps) < 2:
        return None
    steps = [(b - a).total_seconds() for a, b in zip(stamps[:-1], stamps[1:], strict=True)]
    return TimedTrace(
        rec["mas_name"], rec["llm_name"], rec["benchmark_name"], (stamps[-1] - stamps[0]).total_seconds(), steps
    )


PARSERS = {"ChatDev": _chatdev, "OpenManus": _openmanus, "MetaGPT": _metagpt}


def timed_traces(records: list[dict[str, Any]]) -> list[TimedTrace]:
    """Parse every record of a timestamped framework; runs without any step are dropped."""
    out = []
    for rec in records:
        parser = PARSERS.get(rec["mas_name"])
        if parser is None:
            continue
        tt = parser(rec)
        if tt is not None and tt.step_s:
            out.append(tt)
    return out


def ag2_correct(rec: dict[str, Any]) -> bool | None:
    """The ``correct:`` flag of an AG2 trace header (``None`` if absent)."""
    hit = re.search(r"^\s*correct:\s*(True|False)\s*$", rec["trace"]["trajectory"][:5000], re.M)
    return None if hit is None else hit.group(1) == "True"
