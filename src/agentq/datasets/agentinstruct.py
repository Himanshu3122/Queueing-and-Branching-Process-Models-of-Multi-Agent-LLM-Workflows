"""AgentInstruct trajectories (Zeng et al., 2023) as per-attempt workflow records.

Dataset: ``zai-org/AgentInstruct`` on the Hugging Face Hub, 1,866 GPT-4 trajectories on six
AgentBench environments. Each trajectory is a ReAct-style dialogue: the agent (``gpt``) emits a
thought and one action, the environment (``human``) returns an observation.

Mapping to the model of Section 3.1 (one row per agent turn = one LLM call = one *attempt*):

* **tool call**: an agent turn issues one tool call (``n_tool_calls = 1``) unless it is the final
  answer (environment-specific pattern) or the environment reports that no action could be parsed.
  ReAct agents never fork, so the fan-out is 0 or 1.
* **failed attempt**: the environment's next observation is an error (SQL error code, shell error,
  ``Nothing happens.`` in ALFWorld, ``cannot be executed`` in the knowledge-graph tools). The agent
  then re-plans, which is counted as a retry of the same step.
* **step**: a run of failed attempts closed by a successful one. ``steps`` is the number of
  successful attempts.
* **tokens**: context tokens = all dialogue tokens before the call (system prompt, in-context
  example and history), output tokens = the agent turn, both with GPT-4's ``cl100k_base``.

Known limitations, reported with every result: the trajectories were *filtered on task success*
(so task-level failure and the retry cap ``R`` are not observable), there are no timestamps, and
Mind2Web samples are single decisions rather than trajectories (excluded).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentq.datasets.service import TokenLatencyModel

ENVIRONMENTS = ("os", "db", "alfworld", "webshop", "kg")
"""Multi-step environments (Mind2Web is excluded: each sample is one decision)."""

_FINAL = {
    "os": re.compile(r"Act:\s*(answer|finish)", re.I),
    "db": re.compile(r"Action:\s*Answer", re.I),
    "kg": re.compile(r"Final Answer", re.I),
    "alfworld": None,
    "webshop": None,
}
_ERROR = {
    "os": re.compile(
        r"command not found|No such file or directory|syntax error|Permission denied|invalid option|"
        r"unrecognized option|cannot access|Traceback \(most recent call last\)|missing operand"
    ),
    "db": re.compile(r"^\s*\d{4} \([0-9A-Z]{5}\)", re.M),
    "alfworld": re.compile(r"^\s*Nothing happens", re.M),
    "webshop": re.compile(r"Invalid action", re.I),
    "kg": re.compile(r"No executable function found|cannot be executed"),
}
_UNPARSED = re.compile(r"No executable function found")


@dataclass(frozen=True)
class AttemptRecord:
    """One LLM call of a trajectory."""

    step: int
    attempt: int
    failed: bool
    n_tool_calls: int
    context_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class Trajectory:
    """A parsed trajectory: its attempts in execution order."""

    env: str
    traj_id: str
    attempts: tuple[AttemptRecord, ...]
    success: bool = True  # task outcome (AgentInstruct: always True, the data are success-filtered)

    @property
    def steps(self) -> int:
        return sum(1 for a in self.attempts if not a.failed)

    @property
    def n_attempts(self) -> int:
        return len(self.attempts)


def _task_start(env: str, conv: Sequence[dict[str, str]]) -> int:
    """Index of the human turn that poses the actual task (skips instructions and in-context examples)."""
    humans = [i for i, t in enumerate(conv) if t["from"] == "human"]
    if env == "os":
        marks = [i for i in humans if conv[i]["value"].lstrip().startswith("Now, I will start a new problem")]
        return marks[-1] if marks else humans[0]
    if env == "kg":
        marks = [i for i in humans if conv[i]["value"].lstrip().startswith("Question:")]
        return marks[-1] if marks else humans[0]
    if env == "alfworld":
        marks = [i for i in humans if conv[i]["value"].lstrip().startswith("Here is your task")]
        return marks[-1] if marks else humans[0]
    # db, webshop: [instruction, "Ok.", task, ...]
    return humans[1] if len(humans) > 1 else humans[0]


def parse_conversation(
    env: str, conv: Sequence[dict[str, str]], count_tokens: Callable[[str], int], traj_id: str = ""
) -> Trajectory | None:
    """Parse one dialogue into a :class:`Trajectory` (``None`` if it has no agent turn after the task)."""
    if env not in _ERROR:
        raise ValueError(f"unsupported environment '{env}'; choose from {ENVIRONMENTS}")
    start = _task_start(env, conv)
    tokens = [count_tokens(t["value"]) for t in conv]
    final, error = _FINAL[env], _ERROR[env]
    attempts: list[AttemptRecord] = []
    step, attempt = 1, 0
    for i in range(start + 1, len(conv)):
        turn = conv[i]
        if turn["from"] != "gpt":
            continue
        obs = conv[i + 1]["value"] if i + 1 < len(conv) and conv[i + 1]["from"] == "human" else None
        failed = bool(obs is not None and error.search(obs))
        is_final = bool(final is not None and final.search(turn["value"]))
        unparsed = bool(obs is not None and _UNPARSED.search(obs))
        attempts.append(
            AttemptRecord(
                step=step,
                attempt=attempt,
                failed=failed,
                n_tool_calls=0 if (is_final or unparsed) else 1,
                context_tokens=int(sum(tokens[:i])),
                output_tokens=int(tokens[i]),
            )
        )
        if failed:
            attempt += 1
        else:
            step, attempt = step + 1, 0
    if not attempts:
        return None
    return Trajectory(env=env, traj_id=traj_id, attempts=tuple(attempts))


def gpt4_token_counter() -> Callable[[str], int]:
    """Token counter with GPT-4's ``cl100k_base`` encoding (requires ``tiktoken``)."""
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError("install the 'real' extra: pip install -e '.[real]'") from exc
    enc = tiktoken.get_encoding("cl100k_base")
    return lambda text: len(enc.encode(text, disallowed_special=()))


def load_agentinstruct(
    data_dir: str | Path,
    envs: Iterable[str] = ENVIRONMENTS,
    count_tokens: Callable[[str], int] | None = None,
) -> dict[str, list[Trajectory]]:
    """Load and parse ``<data_dir>/<env>.parquet`` for each environment."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install the 'real' extra: pip install -e '.[real]'") from exc
    data_dir = Path(data_dir)
    counter = count_tokens or gpt4_token_counter()
    out: dict[str, list[Trajectory]] = {}
    for env in envs:
        path = data_dir / f"{env}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found; run `python scripts/fetch_datasets.py` first")
        rows = pq.read_table(path).to_pylist()
        trajs = [parse_conversation(env, r["conversations"], counter, str(r.get("id", ""))) for r in rows]
        out[env] = [t for t in trajs if t is not None]
    return out


def to_attempt_rows(trajectories: Iterable[Trajectory], latency: TokenLatencyModel) -> list[dict[str, Any]]:
    """Rows in the schema of :func:`agentq.calibration.calibrate` (tool times are not observed)."""
    rows: list[dict[str, Any]] = []
    for task_id, traj in enumerate(trajectories):
        for a in traj.attempts:
            rows.append(
                {
                    "task_id": task_id,
                    "step": a.step,
                    "attempt": a.attempt,
                    "failed": a.failed,
                    "n_tool_calls": a.n_tool_calls,
                    "llm_service": latency.service_time(a.context_tokens, a.output_tokens),
                    "tool_services": [],
                }
            )
    return rows
