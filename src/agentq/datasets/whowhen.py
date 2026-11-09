"""Who&When (Zhang et al., ICML 2025): failed multi-agent runs annotated with the decisive error.

Source: ``github.com/ag2ai/Agents_Failure_Attribution`` (MIT), folder ``Who&When``, pinned to a
commit in :mod:`agentq.datasets.fetch`. 184 runs, **all of which failed the task**:

* ``Algorithm-Generated`` (126): CaptainAgent expert teams with a code-execution
  ``Computer_terminal`` whose messages carry ``exitcode: N``, so tool failures are observable.
  Histories are truncated at 10 messages, so their lengths are not natural run lengths.
* ``Hand-Crafted`` (58): Magentic-One, where an ``Orchestrator`` delegates (``Orchestrator (->
  X)``) to WebSurfer, FileSurfer, Assistant (coder) and ComputerTerminal.

Each run has ``mistake_agent`` and ``mistake_step`` (0-based index into ``history``): the agent and
step whose error decided the failure. There are no timestamps.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUBSETS = ("Algorithm-Generated", "Hand-Crafted")
_EXIT = re.compile(r"exitcode:\s*(\d+)")


@dataclass(frozen=True)
class Run:
    subset: str
    task_id: str
    history: tuple[dict[str, Any], ...]
    mistake_agent: str
    mistake_step: int

    def speaker(self, i: int) -> str:
        m = self.history[i]
        return str(m.get("name") or m.get("role") or "")

    def tool_outcomes(self) -> list[bool]:
        """Failed (True) / succeeded (False) code executions, in order (Algorithm-Generated)."""
        out = []
        for i, m in enumerate(self.history):
            if self.speaker(i) == "Computer_terminal":
                hit = _EXIT.search(str(m.get("content", "")))
                if hit:
                    out.append(int(hit.group(1)) != 0)
        return out

    def delegations(self) -> list[str]:
        """Targets of the orchestrator's delegations, in order (Hand-Crafted)."""
        out = []
        for m in self.history:
            hit = re.match(r"Orchestrator \(-> (.+)\)", str(m.get("role", "")))
            if hit:
                out.append(hit.group(1))
        return out

    def subagent_turns_per_delegation(self) -> list[int]:
        """Consecutive sub-agent messages between orchestrator messages (Hand-Crafted)."""
        runs, cur = [], 0
        for m in self.history:
            role = str(m.get("role", ""))
            if role.startswith("Orchestrator") or role == "human":
                if cur:
                    runs.append(cur)
                cur = 0
            else:
                cur += 1
        if cur:
            runs.append(cur)
        return runs


def load_whowhen(root: str | Path) -> dict[str, list[Run]]:
    """Load both subsets from ``<root>/Who&When/<subset>/*.json``."""
    base = Path(root) / "Who&When"
    if not base.is_dir():
        raise FileNotFoundError(f"{base} not found; run `python scripts/fetch_datasets.py` first")
    out: dict[str, list[Run]] = {}
    for subset in SUBSETS:
        runs = []
        for path in sorted((base / subset).glob("*.json")):
            d = json.loads(path.read_text(encoding="utf-8"))
            runs.append(
                Run(
                    subset=subset,
                    task_id=str(d.get("question_ID", path.stem)),
                    history=tuple(d["history"]),
                    mistake_agent=str(d["mistake_agent"]),
                    mistake_step=int(d["mistake_step"]),
                )
            )
        out[subset] = runs
    return out
