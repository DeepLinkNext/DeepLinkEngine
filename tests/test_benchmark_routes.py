import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from benchmark.benchmark_io import load
from benchmark.engine_pool import placements
from benchmark.frozen_workload import canonical_mmlu
from benchmark.control_plane import bind_inference_pool


class BenchmarkRoutes(unittest.TestCase):
    def test_full_answer_profile_matches_the_frozen_mmlu_protocol(self):
        root = Path(__file__).resolve().parents[1]
        profile = load(root / "experiments/mmlu-pro-200-out8192.json")
        protocol = load(root / "datasets/mmlu-fixed200.json")
        self.assertEqual(profile["max_tokens"], protocol["generation"]["max_completion_tokens"])
        self.assertEqual(profile["requests"], protocol["tasks_per_arm"])
        self.assertEqual(profile["concurrency"], protocol["tasks_per_arm"])

    def test_prepared_requests_are_the_public_case_order(self):
        with TemporaryDirectory() as name:
            base = Path(name)
            (base / "mmlu-quality018-r2").mkdir()
            (base / "verified-mmlu-quality-inputs.json").write_text('{"cases":[{"case_id":"一"}]}')
            (base / "mmlu-quality018-r2/quality-request-plan.json").write_text('{"requests":[{"model":"m"}]}')
            expected = '{"case_id":"一","body":{"model":"m"}}\n'.encode()
            self.assertEqual(canonical_mmlu(base), expected)

    def test_duplicate_device_on_one_host_is_rejected(self):
        node = {"host":"a", "physical_cards":[0,1], "argv":["engine"], "endpoint":"http://a:1"}
        spec = {"instances":[node, dict(node, endpoint="http://a:2")]}
        with self.assertRaisesRegex(ValueError, "device"):
            placements({}, spec, 2, 2)

    def test_different_hosts_can_use_same_local_card_numbers(self):
        node = {"host":"a", "physical_cards":[0,1], "argv":["engine"], "endpoint":"http://a:1"}
        spec = {"instances":[node, dict(node, host="b", endpoint="http://b:1")]}
        self.assertEqual(len(placements({}, spec, 2, 2)), 2)

    def test_controller_binds_entire_pool(self):
        endpoints = ["http://a:1","http://b:1"]
        config = bind_inference_pool({"endpoint":"http://router:1"}, endpoints)
        self.assertEqual(config["backends"], endpoints)
        with self.assertRaises(ValueError):
            bind_inference_pool({"endpoint":"http://a:1"}, endpoints)

    def test_both_cli_entries_forward_the_same_control_plane_campaign(self):
        import os
        import subprocess
        import sys

        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as name:
            base = Path(name).resolve()
            checkout = base / "controller checkout"
            child = checkout / "experiments/vllm_hust_benchmark/run_vllm_hust_benchmark.py"
            child.parent.mkdir(parents=True)
            child.write_text(
                "import json, os, sys\n"
                "print(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd()}))\n"
                "raise SystemExit(17)\n",
                encoding="utf-8",
            )
            args = ["run", "--scope", "4.1", "--checkpoints", "sealed inputs.json", "--output", "result"]
            env = {**os.environ, "VLLM_HUST_CONTROLLER_ROOT": str(checkout)}
            for script in ("deeplink.py", "run_all.py"):
                with self.subTest(script=script):
                    done = subprocess.run([sys.executable, str(root / script), "control-plane", *args],
                                          cwd=base, env=env, capture_output=True)
                    self.assertEqual(done.returncode, 17, done.stderr)
                    self.assertEqual(json.loads(done.stdout), {"args": args, "cwd": str(base)})

    def test_public_cli_sends_each_request_once_to_its_assigned_engine(self):
        import hashlib
        import socket
        import subprocess
        import sys
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as name:
            base = Path(name)
            sockets = [socket.socket() for _ in range(2)]
            try:
                for item in sockets:
                    item.bind(("127.0.0.1", 0))
                ports = [item.getsockname()[1] for item in sockets]
            finally:
                for item in sockets:
                    item.close()
            nodes = [{"host": "localhost", "physical_cards": [2*i, 2*i+1],
                      "endpoint": f"http://127.0.0.1:{port}",
                      "argv": [sys.executable, str(root/"tests/fixtures/unified_benchmark_server.py"), str(port)]}
                     for i,port in enumerate(ports)]
            rows = [dict(case_id=str(i), body=dict(model="fixture", messages=[], stream=True,
                                                  stream_options={"include_usage":True})) for i in range(4)]
            raw = "\n".join(map(json.dumps,rows)).encode()
            (base/"rows.jsonl").write_bytes(raw)
            (base/"site.json").write_text(json.dumps(dict(requests=str(base/"rows.jsonl"),
                                                          engines={"fixture":{"release":"test","instances":nodes}})))
            (base/"experiment.json").write_text(json.dumps(dict(requests=4, concurrency=4, max_tokens=256,
                                                                request_sha256=hashlib.sha256(raw).hexdigest())))
            common = ["--site", str(base/"site.json"), "--config", str(base/"experiment.json"), "--replicas", "2",
                      "--engine", "fixture", "--route", "direct", "--out", str(base/"run")]
            first = subprocess.check_output([sys.executable, str(root/"deeplink.py"), "benchmark", *common, "--dry-run"])
            second = subprocess.check_output([sys.executable, str(root/"run_all.py"), *common, "--dry-run"])
            self.assertEqual(json.loads(first),json.loads(second))
            subprocess.run([sys.executable, str(root/"deeplink.py"), "benchmark", *common], check=True)
            self.assertEqual(json.loads((base/"run/measurement.json").read_bytes())["successful"], 4)
            self.assertFalse((base/"run/controller-start").exists())
            results = [json.loads(p.read_bytes()) for p in (base/"run").glob("*/result.json")]
            reasoning_only = next(r for r in results if r["case_id"] == "1")
            self.assertEqual(reasoning_only["content"], "")
            self.assertTrue(reasoning_only["reasoning"])
            self.assertIsNone(reasoning_only["first_token"])
            self.assertIsNotNone(reasoning_only["first_output_token"])
            self.assertEqual({r["case_id"]:r["endpoint"] for r in results},
                             {str(i):nodes[i%2]["endpoint"] for i in range(4)})
