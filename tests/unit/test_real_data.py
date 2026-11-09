"""Unit tests for the real-data loaders, the replay simulator and arrival rescaling.

These use small inline fixtures and do not need the downloaded datasets.
"""

from __future__ import annotations

import unittest

import numpy as np

import tests.helpers  # noqa: F401  (puts src/ on sys.path)
from agentq.config import PoolConfig
from agentq.datasets.agentinstruct import parse_conversation, to_attempt_rows
from agentq.datasets.azure import arrival_statistics, rate_surges, rescale_arrivals
from agentq.datasets.azure2024 import hourly_surge_index
from agentq.datasets.bfcl import empirical_pmf, fanout_counts
from agentq.datasets.burstgpt import load_burstgpt, local_surges
from agentq.datasets.mast import timed_traces
from agentq.datasets.mooncake import load_mooncake, prefix_linkage
from agentq.datasets.service import TokenLatencyModel
from agentq.datasets.swegym import extract_record, to_trajectory
from agentq.datasets.whowhen import Run
from agentq.queueing import lognormal_params
from agentq.sim import poisson_arrivals, simulate_chain, simulate_replay
from tests.helpers import profiles


def words(text: str) -> int:
    return len(text.split())


def turn(who: str, text: str) -> dict[str, str]:
    return {"from": who, "value": text}


DB_CONV = [
    turn("human", "instructions for the database task"),
    turn("gpt", "Ok."),
    turn("human", "How many rows are in table t?"),
    turn("gpt", "Action: Operation SELECT COUNT(*) FROM tt"),
    turn("human", "1146 (42S02): Table 'x.tt' doesn't exist"),
    turn("gpt", "Action: Operation SELECT COUNT(*) FROM t"),
    turn("human", "[(4,)]"),
    turn("gpt", "Action: Answer Final Answer: [4]"),
]

KG_CONV = [
    turn("human", "tool instructions"),
    turn("gpt", "understood"),
    turn("human", "Question: example question"),  # in-context example, must be skipped
    turn("gpt", "Action: get_relations(a)"),
    turn("human", "Observation: [r1]"),
    turn("gpt", "Final Answer: #0"),
    turn("human", "Question: real question"),
    turn("gpt", "Action: get_relation(b)"),
    turn("human", "No executable function found! Need to recheck the action."),
    turn("gpt", "Action: get_relations(b)"),
    turn("human", "Observation: [r2]"),
    turn("gpt", "Final Answer: #1"),
]


class TestAgentInstructParser(unittest.TestCase):
    def test_db_failure_retry_and_final_answer(self) -> None:
        t = parse_conversation("db", DB_CONV, words)
        assert t is not None
        self.assertEqual([a.failed for a in t.attempts], [True, False, False])
        self.assertEqual([(a.step, a.attempt) for a in t.attempts], [(1, 0), (1, 1), (2, 0)])
        self.assertEqual([a.n_tool_calls for a in t.attempts], [1, 1, 0])
        self.assertEqual(t.steps, 2)
        # context = all words before the call; output = the call itself
        self.assertEqual(t.attempts[0].context_tokens, sum(words(x["value"]) for x in DB_CONV[:3]))
        self.assertEqual(t.attempts[0].output_tokens, words(DB_CONV[3]["value"]))

    def test_kg_skips_in_context_example_and_unparsed_action_has_no_tool(self) -> None:
        t = parse_conversation("kg", KG_CONV, words)
        assert t is not None
        self.assertEqual(t.n_attempts, 3)
        self.assertEqual([a.failed for a in t.attempts], [True, False, False])
        self.assertEqual([a.n_tool_calls for a in t.attempts], [0, 1, 0])

    def test_rows_match_calibration_schema(self) -> None:
        t = parse_conversation("db", DB_CONV, words)
        assert t is not None
        rows = to_attempt_rows([t, t], TokenLatencyModel())
        self.assertEqual(len(rows), 6)
        self.assertEqual({r["task_id"] for r in rows}, {0, 1})
        self.assertTrue(all(r["llm_service"] > 0 and r["tool_services"] == [] for r in rows))

    def test_unknown_environment(self) -> None:
        with self.assertRaises(ValueError):
            parse_conversation("mind2web", DB_CONV, words)


class TestLatencyModel(unittest.TestCase):
    def test_linear(self) -> None:
        m = TokenLatencyModel(overhead_s=0.5, prefill_ms_per_token=1.0, decode_ms_per_token=10.0)
        self.assertAlmostEqual(m.service_time(1000, 100), 0.5 + 1.0 + 1.0)
        with self.assertRaises(ValueError):
            TokenLatencyModel(decode_ms_per_token=0)


class TestArrivals(unittest.TestCase):
    def test_rescale_hits_target_rate_and_keeps_shape(self) -> None:
        rng = np.random.default_rng(0)
        t = np.sort(rng.uniform(0, 1000, 5000))
        for offset in (0.0, 0.37):
            r = rescale_arrivals(t, rate=2.0, offset_fraction=offset)
            self.assertAlmostEqual((r.size - 1) / r[-1], 2.0, places=6)
            self.assertTrue(np.all(np.diff(r) >= 0))
            self.assertEqual(r[0], 0.0)
        self.assertEqual(rescale_arrivals(t, 1.0, n=100).size, 100)

    def test_poisson_statistics_near_one(self) -> None:
        st = arrival_statistics(poisson_arrivals(np.random.default_rng(1), 5.0, 50_000))
        self.assertAlmostEqual(st["interarrival_scv"], 1.0, delta=0.05)
        self.assertAlmostEqual(st["index_of_dispersion"]["10.0"], 1.0, delta=0.15)


class TestMooncake(unittest.TestCase):
    def _write(self, rows: list[dict]) -> str:
        import json
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
            fh.write("\n".join(json.dumps(r) for r in rows))
        return fh.name

    def test_slot_spreading_and_tokens(self) -> None:
        rows = [
            {"timestamp": 0, "input_length": 600, "output_length": 5, "hash_ids": [0, 1]},
            {"timestamp": 0, "input_length": 700, "output_length": 6, "hash_ids": [0, 2]},
            {"timestamp": 3000, "input_length": 1100, "output_length": 7, "hash_ids": [0, 2, 3]},
        ]
        t, tok = load_mooncake(self._write(rows))
        np.testing.assert_allclose(t, [0.0, 1.5, 3.75])  # mid-slot positions 0.75, 2.25, 4.5, shifted to start at 0
        self.assertEqual(tok["output_tokens"].tolist(), [5, 6, 7])

    def test_prefix_linkage_links_extension(self) -> None:
        rows = [
            {"timestamp": 0, "input_length": 1100, "output_length": 5, "hash_ids": [7, 8, 9]},
            {"timestamp": 3000, "input_length": 1600, "output_length": 5, "hash_ids": [7, 8, 10, 11]},
            {"timestamp": 6000, "input_length": 600, "output_length": 5, "hash_ids": [20, 21]},
        ]
        link = prefix_linkage(self._write(rows))
        self.assertAlmostEqual(link["linked_fraction"], 1 / 3)
        self.assertAlmostEqual(link["median_gap_to_parent_s"], 3.0)


class TestBfcl(unittest.TestCase):
    def test_fanout_counts_and_pmf(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            pa = Path(d) / "possible_answer"
            pa.mkdir()
            rows = [{"id": "a", "ground_truth": [{"f": {}}]}, {"id": "b", "ground_truth": [{"f": {}}, {"g": {}}]}]
            (pa / "BFCL_v3_live_parallel.json").write_text("\n".join(json.dumps(r) for r in rows))
            counts = fanout_counts(d, ("live_parallel",))
        self.assertEqual(counts.tolist(), [1, 2])
        self.assertEqual(empirical_pmf(np.array([1, 1, 2, 4])), {1: 0.5, 2: 0.25, 4: 0.25})


class TestWhoWhen(unittest.TestCase):
    def test_tool_outcomes_and_delegations(self) -> None:
        alg = Run(
            "Algorithm-Generated",
            "t",
            (
                {"name": "Coder", "content": "run"},
                {"name": "Computer_terminal", "content": "exitcode: 1 (execution failed)"},
                {"name": "Computer_terminal", "content": "no exit code here"},
                {"name": "Computer_terminal", "content": "exitcode: 0 (execution succeeded)"},
            ),
            "Coder",
            0,
        )
        self.assertEqual(alg.tool_outcomes(), [True, False])
        hand = Run(
            "Hand-Crafted",
            "h",
            (
                {"role": "human", "content": "q"},
                {"role": "Orchestrator (thought)", "content": "plan"},
                {"role": "Orchestrator (-> WebSurfer)", "content": "search"},
                {"role": "WebSurfer", "content": "result"},
                {"role": "Orchestrator (-> Assistant)", "content": "code"},
                {"role": "Assistant", "content": "done"},
            ),
            "WebSurfer",
            3,
        )
        self.assertEqual(hand.delegations(), ["WebSurfer", "Assistant"])
        self.assertEqual(hand.subagent_turns_per_delegation(), [1, 1])


class TestSwegym(unittest.TestCase):
    @staticmethod
    def _call(name: str, args: str = "{}") -> dict:
        return {"function": {"name": name, "arguments": args}}

    def test_extract_record(self) -> None:
        c = self._call
        row = {
            "run_id": "gpt-4o-2024-08-06_maxiter_30_N_v2.1-no-hint-train-t04-run_1",
            "instance_id": "x__y-1",
            "resolved": True,
            "test_result": {"report": {"empty_generation": False}},
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "fix the bug"},
                {"role": "assistant", "content": "", "tool_calls": [c("str_replace_editor"), c("execute_bash")]},
                {"role": "tool", "content": "OBSERVATION: ERROR: No replacement was performed"},
                {"role": "tool", "content": "OBSERVATION: ok\n[Command finished with exit code 1]"},
                {"role": "assistant", "content": "", "tool_calls": [c("execute_bash")]},
                {"role": "tool", "content": "OBSERVATION: bash: foo: command not found\n[... exit code 127]"},
                {"role": "assistant", "content": "done", "tool_calls": [c("finish")]},
            ],
        }
        rec = extract_record(row, words)
        self.assertEqual(rec["model"], "gpt-4o-2024-08-06")
        self.assertEqual((rec["maxiter"], rec["temperature"]), (30, 0.4))
        self.assertEqual(rec["ending"], "finish")
        # [context, output, K (finish excluded), strict, broad]
        self.assertEqual([a[2:] for a in rec["attempts"]], [[2, 1, 1], [1, 1, 1], [0, 0, 0]])
        self.assertEqual(rec["attempts"][0][0], 4)  # context = "sys" (1 word) + "fix the bug" (3 words)
        t = to_trajectory(rec)
        self.assertTrue(t.success)
        self.assertEqual([(a.step, a.attempt) for a in t.attempts], [(1, 0), (1, 1), (1, 2)])

    def test_abort_and_tail_repeats(self) -> None:
        c = self._call
        same = {"role": "assistant", "content": "", "tool_calls": [c("str_replace_editor", '{"command":"view"}')]}
        obs = {"role": "tool", "content": "OBSERVATION: listing"}
        row = {
            "run_id": "claude-3-5-sonnet-20241022_maxiter_50_N_v2.1-no-hint-run_1",
            "instance_id": "i",
            "resolved": False,
            "test_result": {"report": {"empty_generation": True}},
            "messages": [{"role": "user", "content": "task"}, same, obs, same, obs, same, obs],
        }
        rec = extract_record(row, words)
        self.assertEqual((rec["ending"], rec["tail_repeats"], rec["empty_patch"]), ("abort", 3, True))
        self.assertIsNone(rec["temperature"])


class TestBurstgpt(unittest.TestCase):
    def test_load_spreads_ties_and_flags_failures(self) -> None:
        import tempfile
        from pathlib import Path

        csv = (
            "Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type\n"
            "10,ChatGPT,5,3,8,API log\n10,GPT-4,7,0,7,API log\n12,ChatGPT,1,1,2,Conversation log\n"
        )
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "b.csv"
            path.write_text(csv)
            df = load_burstgpt(path)
        np.testing.assert_allclose(df["time"], [0.0, 0.5, 2.25])  # 10.25, 10.75, 12.5 shifted to start at 0
        self.assertEqual(df["failed"].tolist(), [False, True, False])

    def test_local_surges(self) -> None:
        steady = np.arange(0, 7200, 1.0)
        self.assertEqual(local_surges(steady)["fraction_above"], 0.0)
        burst = np.sort(np.concatenate([steady, np.arange(0, 300, 0.5)]))  # first 5 minutes at 3x
        self.assertGreater(local_surges(burst)["fraction_above"], 0.0)


class TestMastParsers(unittest.TestCase):
    @staticmethod
    def _rec(mas: str, text: str) -> dict:
        return {"mas_name": mas, "llm_name": "GPT-4o", "benchmark_name": "b", "trace": {"trajectory": text}}

    def test_chatdev_calls_tokens_and_status(self) -> None:
        log = (
            "[2025-31-03 19:09:41 INFO] start\n"
            "[2025-31-03 19:09:41 INFO] flask app.py did not start for online log\n"
            '[2025-31-03 19:09:51 INFO] HTTP Request: POST https://api/v1 "HTTP/1.1 200 OK"\n'
            "[2025-31-03 19:09:51 INFO] **[OpenAI_Usage_Info Receive]**\nprompt_tokens: 486\ncompletion_tokens: 69\n\n"
            "[2025-31-03 19:09:52 INFO] flask app.py did not start for online log\n"
            '[2025-31-03 19:09:55 INFO] HTTP Request: POST https://api/v1 "HTTP/1.1 502 Bad"\n'
            '[2025-31-03 19:09:59 INFO] HTTP Request: POST https://api/v1 "HTTP/1.1 200 OK"\n'
            "[2025-31-03 19:09:59 INFO] **[OpenAI_Usage_Info Receive]**\nprompt_tokens: 556\ncompletion_tokens: 48\n\n"
            "[2025-31-03 19:10:01 INFO] done\n"
        )
        t = timed_traces([self._rec("ChatDev", log)])[0]
        self.assertEqual(t.step_s, [10.0, 4.0])
        self.assertEqual(t.tokens, [(486, 69), (556, 48)])
        self.assertEqual(t.http_status, {"200": 2, "502": 1})
        self.assertEqual(t.total_s, 20.0)

    def test_metagpt_and_openmanus(self) -> None:
        mg = (
            "[2025-03-31 12:53:42] FROM: Human TO: all\nx\n"
            "[2025-03-31 12:54:14] NEW MESSAGES:\ny\n[2025-03-31 12:54:30] NEW MESSAGES:\n"
        )
        om = (
            "2025-04-01 01:27:44.445 | INFO     | plan - Creating initial plan\n"
            "2025-04-01 01:27:47.937 | INFO     | app.llm - Token usage: Input=426, Completion=116\n"
        )
        g = timed_traces([self._rec("MetaGPT", mg), self._rec("OpenManus", om), self._rec("AG2", "no stamps")])
        self.assertEqual([t.mas for t in g], ["MetaGPT", "OpenManus"])
        self.assertEqual(g[0].step_s, [32.0, 16.0])
        self.assertAlmostEqual(g[1].step_s[0], 3.492)
        self.assertEqual(g[1].tokens, [(426, 116)])


class TestAzure2024(unittest.TestCase):
    def test_hourly_surge_index(self) -> None:
        steady = np.arange(0, 7200, 1.0)  # two hours at 1/s
        burst = np.arange(3600, 3660, 0.25)  # first minute of hour 2 at 5/s on top
        idx = hourly_surge_index(np.sort(np.concatenate([steady, burst])))
        self.assertEqual(idx["hour"].tolist(), [0, 1])
        self.assertEqual(idx["share_above"][0], 0.0)
        self.assertAlmostEqual(idx["share_above"][1], 1 / 60)
        self.assertGreater(idx["peak_ratio"][1], 4.0)


class TestRateSurges(unittest.TestCase):
    def test_constant_rate_has_no_surges(self) -> None:
        r = rate_surges(np.arange(0, 3600, 0.25))
        self.assertAlmostEqual(r["windows"]["300.0"]["peak_to_mean"], 1.0, places=2)
        self.assertEqual(r["windows"]["300.0"]["fraction_above"], 0.0)

    def test_doubled_rate_window_is_detected(self) -> None:
        t = np.sort(np.concatenate([np.arange(0, 3000, 1.0), np.arange(3000, 3600, 0.5)]))
        r = rate_surges(t)
        self.assertGreater(r["windows"]["300.0"]["fraction_above"], 0.0)


class TestReplaySimulator(unittest.TestCase):
    def test_replay_of_model_distributed_tasks_matches_chain_simulator(self) -> None:
        """Tasks drawn from the model's own law, replayed, must reproduce simulate_chain statistics."""
        base = profiles()["P3"]
        system = base.with_updates(**{"workflow.fanout_kind": "fixed", "workflow.fanout": 1})
        wf, rate, n = system.workflow, 0.25, 8000
        rng = np.random.default_rng(3)
        mu, sg = lognormal_params(system.llm.mean_service, system.llm.cv)
        tasks = []
        for _ in range(n):
            attempts, ok = [], True
            for _step in range(int(rng.geometric(wf.q))):
                for a in range(wf.max_retries + 1):
                    failed = rng.random() < wf.failure_prob
                    attempts.append((float(rng.lognormal(mu, sg)), 1))
                    if not failed:
                        break
                    if a == wf.max_retries:
                        ok = False
                if not ok:
                    break
            tasks.append((tuple(attempts), ok))
        arr = poisson_arrivals(rng, rate, n)
        rep = simulate_replay(tasks, arr, system.llm.servers, system.tool, seed=1)
        ref = simulate_chain(system, rate, n, seed=1)
        self.assertAlmostEqual(rep.mean / ref.mean, 1.0, delta=0.08)
        self.assertAlmostEqual(rep.success_prob, ref.success_prob, delta=0.02)
        self.assertAlmostEqual(rep.rho_llm / ref.rho_llm, 1.0, delta=0.08)

    def test_single_task_no_queueing_is_sum_of_services(self) -> None:
        tool = PoolConfig(servers=4, mean_service=1e-9, cv=0.1)
        tasks = [((((2.0, 0), (3.0, 1))), True)] * 20
        arr = np.arange(20) * 100.0
        res = simulate_replay(tasks, arr, 1, tool, warmup_fraction=0.0)
        self.assertAlmostEqual(res.mean, 5.0, places=5)
        self.assertEqual(res.success_prob, 1.0)
        self.assertAlmostEqual(res.llm_calls_per_task, 2.0)

    def test_input_validation(self) -> None:
        tool = PoolConfig(servers=1, mean_service=1.0, cv=1.0)
        with self.assertRaises(ValueError):
            simulate_replay([(((1.0, 0),), True)] * 5, np.arange(20.0), 1, tool)


if __name__ == "__main__":
    unittest.main()
