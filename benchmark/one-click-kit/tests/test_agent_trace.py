"""Synthetic protocol/transport regressions, never benchmark measurements."""

import argparse
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import agent_trace as trace
import dataset_client
import third_party


class AgentTraceTests(unittest.TestCase):
    def test_swe_selection_is_repository_proportional_and_order_independent(self):
        cases = [
            f"instance_{repo}__repo-{index:040x}"
            for repo, count in (("large", 75), ("small", 25))
            for index in range(count)
        ]
        chosen, quota = trace.select_tasks("swe", cases)
        self.assertEqual(quota, {"large__repo": 24, "small__repo": 8})
        self.assertEqual(len(chosen), 32)
        self.assertEqual(
            (chosen, quota), trace.select_tasks("swe", list(reversed(cases)))
        )

    def test_terminal_selection_does_not_depend_on_input_order(self):
        cases = ["synthetic-task-" + str(i) for i in range(89)]
        chosen, quota = trace.select_tasks("terminal", cases)
        self.assertIsNone(quota)
        self.assertEqual(len(set(chosen)), 32)
        self.assertEqual((chosen, quota), trace.select_tasks("terminal", cases[::-1]))

    def test_requests_preserve_history_and_do_not_truncate_long_inputs(self):
        candidates = [
            {
                "source_index": index,
                "body": {"messages": [{"role": "user", "content": text}]},
            }
            for index, text in enumerate(("first", "long", "last", "last"))
        ]
        frozen = copy.deepcopy(candidates)
        selected, audit = trace.select_requests(
            candidates,
            lambda body: 20000 if body["messages"][0]["content"] == "long" else 120,
        )
        self.assertEqual([s["source_index"] for s in selected], [0, 2])
        self.assertEqual(
            [e["reason"] for e in audit["excluded"]],
            ["prompt_over16384", "duplicate_request_body"],
        )
        self.assertEqual(candidates, frozen)

    def test_lone_request_is_never_duplicated_to_fill_quota(self):
        candidate = {
            "source_index": 0,
            "body": {"messages": [{"role": "user", "content": "synthetic"}]},
        }
        chosen, _ = trace.select_requests([candidate], lambda _: 12)
        self.assertEqual(len(chosen), 1)
        chosen, audit = trace.select_requests([candidate], lambda _: 18000)
        self.assertEqual(chosen, [])
        self.assertEqual(audit["eligible_requests"], 0)

    def test_redacted_source_history_is_excluded_without_filling_messages(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "synthetic.traj"
            path.write_text(
                json.dumps(
                    {
                        "trajectory": [
                            {"query": [{}, {"role": "user", "content": "synthetic"}]}
                        ]
                    }
                )
            )
            candidates = trace.extract("swe", path)
            chosen, audit = trace.select_requests(
                candidates,
                lambda _: self.fail("Do not fabricate a missing system message"),
            )
            self.assertEqual(chosen, [])
            self.assertEqual(
                audit["excluded"][0]["reason"], "incomplete_published_message"
            )

    def test_swe_tool_results_keep_original_call_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "synthetic.traj"
            path.write_text(
                json.dumps(
                    {
                        "trajectory": [
                            {
                                "query": [
                                    {
                                        "role": "tool",
                                        "content": "synthetic observed output",
                                        "tool_call_ids": ["original-tool-id"],
                                    }
                                ]
                            }
                        ]
                    }
                )
            )
            body = trace.extract("swe", path)[0]["body"]
            self.assertEqual(
                body["messages"],
                [
                    {
                        "role": "tool",
                        "content": "synthetic observed output",
                        "tool_call_id": "original-tool-id",
                    }
                ],
            )

    def test_template_counting_decodes_arguments_without_editing_wire_history(self):
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "bash",
                                "arguments": '{"command":"synthetic"}',
                            }
                        }
                    ],
                }
            ]
        }
        messages = trace.template_messages(body)
        self.assertEqual(
            messages[0]["tool_calls"][0]["function"]["arguments"],
            {"command": "synthetic"},
        )
        self.assertIsInstance(
            body["messages"][0]["tool_calls"][0]["function"]["arguments"], str
        )
        body["messages"][0]["tool_calls"][0]["function"]["arguments"] = (
            "malformed original JSON"
        )
        with self.assertRaises(trace.InvalidPublishedRequest):
            trace.template_messages(body)

    def test_unexpected_target_tokenization_is_failed_and_retains_the_response(self):
        from http.server import ThreadingHTTPServer
        import threading
        from test_third_party_execution import TransportHandler

        server = ThreadingHTTPServer(("127.0.0.1", 0), TransportHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                out = Path(temp)
                task = {
                    "case_id": "synthetic-trace",
                    "source_body": {
                        "messages": [{"role": "user", "content": "synthetic"}]
                    },
                    "source_body_sha256": "synthetic",
                    "prompt_tokens": 13,
                }
                contract = {
                    "base_url": "http://127.0.0.1:" + str(server.server_port),
                    "timeout": 2,
                    "concurrency": 1,
                    "model": "synthetic",
                    "generation": {"max_tokens": 2},
                }
                with (
                    patch(
                        "agent_trace.checked",
                        return_value={"dataset": "terminal", "plan": "synthetic"},
                    ),
                    patch("agent_trace.load_plan", return_value=(contract, [task])),
                    patch("agent_trace.admit_tokenization"),
                ):
                    self.assertEqual(trace.run("synthetic", out), 2)
                record = json.loads(
                    (out / "evaluation/requests/synthetic-trace.json").read_bytes()
                )
                self.assertEqual(record["status"], "tokenization_error")
                self.assertEqual(record["content"], "red")
                self.assertTrue(Path(record["stream_capture"]).exists())
                self.assertEqual(
                    json.loads((out / "evaluation/phase.json").read_bytes())[
                        "successful"
                    ],
                    0,
                )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_images_are_explicitly_outside_text_trace_profile(self):
        candidate = {
            "source_index": 7,
            "body": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "synthetic"}}
                        ],
                    }
                ]
            },
        }
        chosen, audit = trace.select_requests(
            [candidate], lambda _: self.fail("Must not silently tokenize/drop image")
        )
        self.assertEqual(chosen, [])
        self.assertEqual(audit["excluded"][0]["reason"], "non_text_content")

    def test_swe_uses_recorded_query_and_ignores_saved_output_and_score(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "synthetic.traj"
            query = [
                {"role": "system", "content": "synthetic system", "agent": "main"},
                {"role": "user", "content": "synthetic task", "message_type": "task"},
            ]
            path.write_text(
                json.dumps(
                    {
                        "trajectory": [
                            {
                                "query": query,
                                "response": "synthetic saved answer",
                                "observation": "synthetic after-query output",
                            }
                        ],
                        "info": {"reward": 1},
                    }
                )
            )
            first = trace.extract("swe", path)
            self.assertEqual(
                first[0]["body"],
                {
                    "messages": [
                        {"role": "system", "content": "synthetic system"},
                        {"role": "user", "content": "synthetic task"},
                    ]
                },
            )
            data = json.loads(path.read_bytes())
            data["info"]["reward"] = 0
            data["trajectory"][0]["response"] = "changed saved answer"
            path.write_text(json.dumps(data))
            self.assertEqual(trace.extract("swe", path), first)

    def test_terminal_keeps_exact_tools_calls_results_and_timestamp(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "synthetic.jsonl"
            body = {
                "model": "historical-model",
                "max_tokens": 32768,
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "original-call",
                                "type": "function",
                                "function": {"name": "bash", "arguments": "{}"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "original-call",
                        "content": "historical tool output",
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "bash", "parameters": {"type": "object"}},
                    }
                ],
            }
            path.write_text(json.dumps({"t": 123, "pid": 42, "payload": body}) + "\n")
            row = trace.extract("terminal", path)[0]
            self.assertEqual(row["body"], {k: body[k] for k in ("messages", "tools")})
            self.assertEqual(row["original_timestamp_ms"], 123)
            self.assertNotIn("max_tokens", row["body"])

    def test_changed_cached_source_is_rejected_without_network(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "source.json").write_text("changed synthetic source")
            with patch("urllib.request.urlopen") as network:
                with self.assertRaisesRegex(
                    ValueError, "Cached public trajectory changed"
                ):
                    trace.fetch({"cache_key": "source.json", "sha256": "0" * 64}, root)
            network.assert_not_called()

    def test_new_request_uses_frozen_budget_and_unique_no_reuse_salt(self):
        contract = {
            "generation": {
                "max_tokens": 2048,
                "temperature": 0,
                "seed": 17,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            "model": "target-model",
        }
        task = {
            "source_body": {
                "messages": [{"role": "user", "content": "synthetic"}],
                "tools": [],
            }
        }
        one = trace.request_body(contract, task)
        two = trace.request_body(contract, task)
        self.assertNotEqual(one["cache_salt"], two["cache_salt"])
        self.assertEqual(one["messages"], task["source_body"]["messages"])
        self.assertEqual(one["max_tokens"], 2048)
        self.assertEqual(one["stream_options"], {"include_usage": True})
        self.assertNotIn("cache_salt", task["source_body"])

    def test_trace_checked_dispatch_does_not_enter_quality_client(self):
        with tempfile.TemporaryDirectory() as temp:
            spec = Path(temp) / "spec.json"
            spec.write_text('{"dataset":"terminal"}')
            with (
                patch(
                    "agent_trace.checked", return_value={"dataset": "terminal"}
                ) as checked,
                patch("dataset_client.client") as quality,
            ):
                data, module = dataset_client.checked(spec)
            self.assertEqual(data["dataset"], "terminal")
            self.assertIs(module, trace)
            checked.assert_called_once()
            quality.assert_not_called()

    def test_frozen_source_protocol_and_all_selected_requests_are_consistent(self):
        pin = trace.protocol()
        for name, data in pin["datasets"].items():
            selected, quota = trace.select_tasks(
                name, data["available_published_task_ids"]
            )
            self.assertEqual(selected, data["scope"]["selected_task_ids"])
            self.assertEqual(quota, data["scope"]["repository_quotas"])
            self.assertLessEqual(len(data["requests"]), 64)
            self.assertEqual(
                len(data["requests"]), len({r["case_id"] for r in data["requests"]})
            )
            self.assertTrue(all(r["prompt_tokens"] <= 16384 for r in data["requests"]))
            self.assertTrue(all("source_body" not in r for r in data["requests"]))
            self.assertFalse(data["scope"]["quality_measured"])

    def test_stages_add_both_trace_arms_without_judges(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "plans").mkdir()
            args = argparse.Namespace(
                devices="2,3",
                physical_devices="5,6",
                model=root,
                report_python=None,
                continuous_evidence=None,
                smoke_evaluators=False,
            )
            frozen = {
                "optimized_deployment": str(root),
                "baseline_deployment": str(root),
                "datasets": ["swe", "terminal"],
            }
            with patch("third_party.descriptor", return_value={}) as descriptor:
                stages = third_party.build_stages(args, root, frozen)
            self.assertEqual(
                [s["name"] for s in stages],
                [
                    "continuous-B0",
                    "continuous-B1",
                    "swe-trace-B0",
                    "swe-trace-B1",
                    "terminal-trace-B0",
                    "terminal-trace-B1",
                ],
            )
            self.assertEqual(descriptor.call_count, 4)
            self.assertTrue(
                all(call.args[3] == "trace" for call in descriptor.call_args_list)
            )

    def test_actual_tokenization_mismatch_stops_before_model_generation(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            task = {
                "case_id": "synthetic",
                "prompt_tokens": 12,
                "source_body": {"messages": [{"role": "user", "content": "synthetic"}]},
            }
            contract = {
                "base_url": "http://127.0.0.1:1",
                "model": "synthetic",
                "generation": {"chat_template_kwargs": {"enable_thinking": False}},
            }
            opener = SimpleNamespace(
                open=lambda *a, **k: io.BytesIO(
                    json.dumps({"count": 13, "tokens": list(range(13))}).encode()
                )
            )
            with (
                patch(
                    "agent_trace.checked",
                    return_value={
                        "dataset": "swe",
                        "plan": "synthetic",
                        "service_role": "B0",
                    },
                ),
                patch("agent_trace.load_plan", return_value=(contract, [task])),
                patch("urllib.request.build_opener", return_value=opener),
                patch("streaming_transport.call") as generate,
            ):
                with self.assertRaisesRegex(
                    ValueError, "Actual service tokenization differs"
                ):
                    trace.run("synthetic", out)
            generate.assert_not_called()
            admission = json.loads(
                (out / "trace-tokenizer-admission.json").read_bytes()
            )
            self.assertFalse(admission["complete"])
            self.assertEqual(admission["generation_requests"], 0)
            self.assertTrue((out / "tokenize/synthetic.json.gz").exists())

    def test_actual_tokenization_qualifies_and_captures_exact_token_ids(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            tokens = [1, 2, 3]
            task = {
                "case_id": "synthetic",
                "prompt_tokens": 3,
                "source_body": {
                    "messages": [{"role": "user", "content": "synthetic"}],
                    "tools": [],
                },
            }
            contract = {
                "base_url": "http://127.0.0.1:1",
                "model": "synthetic",
                "generation": {"chat_template_kwargs": {"enable_thinking": False}},
            }
            opener = SimpleNamespace(
                open=lambda *a, **k: io.BytesIO(
                    json.dumps({"count": len(tokens), "tokens": tokens}).encode()
                )
            )
            with patch("urllib.request.build_opener", return_value=opener):
                admission = trace.admit_tokenization(
                    {"service_role": "B0"}, contract, [task], out
                )
            self.assertTrue(admission["complete"])
            self.assertEqual(
                admission["inputs"]["synthetic"]["token_ids_sha256"],
                trace.digest(tokens),
            )
            self.assertEqual(admission["generation_requests"], 0)

    def test_b1_rejects_equal_count_but_different_actual_input_tokens(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prior = root / "B0"
            (prior / "service").mkdir(parents=True)
            trace.save(prior / "receipt.json", {"completed": True})
            trace.save(
                prior / "service/trace-tokenizer-admission.json",
                {
                    "complete": True,
                    "inputs": {
                        "synthetic": {"token_ids_sha256": trace.digest([1, 2, 3])}
                    },
                },
            )
            task = {
                "case_id": "synthetic",
                "prompt_tokens": 3,
                "source_body": {"messages": [{"role": "user", "content": "synthetic"}]},
            }
            contract = {
                "base_url": "http://127.0.0.1:1",
                "model": "synthetic",
                "generation": {"chat_template_kwargs": {"enable_thinking": False}},
            }
            opener = SimpleNamespace(
                open=lambda *a, **k: io.BytesIO(
                    json.dumps({"count": 3, "tokens": [1, 2, 4]}).encode()
                )
            )
            data = {
                "service_role": "B1",
                "baseline_trace_run": str(prior),
                "contract_sha256": "synthetic",
            }
            with (
                patch("qualification.sealed_files"),
                patch(
                    "agent_trace.checked",
                    return_value={"service_role": "B0", "contract_sha256": "synthetic"},
                ),
                patch("urllib.request.build_opener", return_value=opener),
            ):
                with self.assertRaisesRegex(
                    ValueError, "actual input token IDs differ"
                ):
                    trace.admit_tokenization(data, contract, [task], root / "B1")

    def test_trace_report_does_not_describe_cpu_connection_as_a_blocker(self):
        from third_party_report import targets

        result = targets(None, None, {"datasets": ["swe", "terminal"]}, {})["3.4.3"]
        self.assertIn("已接入", result["status"])
        self.assertIn("不测任务解决率", result["reason"])
        self.assertFalse(result["accepted"])


if __name__ == "__main__":
    unittest.main()
