"""Synthetic transport and workflow regressions; these are not model results."""

import argparse
import gzip
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import dataset_client
import dataset_contract
import resource_guard
from restore_b0_image import extract_layer
from streaming_transport import call
import third_party
from third_party_report import targets, verify_campaign_bindings, read_bound


class TransportHandler(BaseHTTPRequestHandler):
    mode = "good"

    def log_message(self, *_):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if self.mode == "error":
            self.wfile.write(
                b'data: {"error":{"message":"synthetic stream failure"}}\n\n'
            )
            return
        events = [
            {"choices": [{"delta": {"content": "red"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "length"}]},
            {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 2}},
        ]
        for event in events:
            self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
        if self.mode != "partial":
            self.wfile.write(b"data: [DONE]\n\n")


class ThirdPartyExecutionTests(unittest.TestCase):
    def test_unbound_new_summary_cannot_supply_a_score(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary = root / "summary.json"
            summary.write_text('{"synthetic_score":1}')
            with self.assertRaisesRegex(ValueError, "no campaign binding"):
                read_bound(summary, root, {})

    def test_parent_seal_detects_changed_dataset_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            quality = root / "frontier-quality-B0"
            quality.mkdir()
            summary = quality / "summary.json"
            summary.write_text('{"synthetic_score":0}')
            seal = quality / "checksums.json"
            seal.write_text(
                json.dumps(
                    {"summary.json": hashlib.sha256(summary.read_bytes()).hexdigest()}
                )
            )
            (root / "campaign-checksums.json").write_text(
                json.dumps(
                    {
                        "frontier-quality-B0/checksums.json": hashlib.sha256(
                            seal.read_bytes()
                        ).hexdigest()
                    }
                )
            )
            verify_campaign_bindings(root)
            summary.write_text('{"synthetic_score":1}')
            with self.assertRaisesRegex(ValueError, "child content changed"):
                verify_campaign_bindings(root)

    def test_interrupted_runtime_restore_cannot_publish_partial_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp)

            def interrupted(path):
                path.mkdir()
                (path / "partial").write_text("synthetic interrupted extraction")
                raise RuntimeError("synthetic interruption")

            with self.assertRaises(RuntimeError):
                third_party.cached_runtime(cache, "sdk", None, interrupted)
            self.assertFalse((cache / "sdk").exists())
            self.assertEqual(len(list(cache.glob("sdk-attempt-*/partial"))), 1)

            def complete(path):
                path.mkdir()
                (path / "ready").write_text("synthetic complete runtime")

            ready = third_party.cached_runtime(cache, "sdk", None, complete)
            self.assertTrue((ready / "ready").is_file())
            self.assertEqual((cache / "sdk").resolve(), ready)
            self.assertEqual(
                third_party.cached_runtime(cache, "sdk", None, interrupted), ready
            )

    def test_new_failed_pair_cannot_inherit_historical_pass(self):
        historical = {
            "formal_targets": {
                "3.4.1": {"accepted": True, "display_value": "synthetic-reference"}
            }
        }
        current = {
            "formal_targets": {
                key: {
                    "accepted": False,
                    "display_value": "synthetic-failed",
                    "reason": "synthetic failed numeric threshold",
                }
                for key in ("3.4.2", "3.4.4")
            }
        }
        result = targets(historical, current, {}, {})
        self.assertTrue(result["3.4.1"]["accepted"])
        self.assertFalse(result["3.4.2"]["accepted"])
        self.assertFalse(result["3.4.3"]["accepted"])
        self.assertFalse(result["3.4.4"]["accepted"])
        self.assertIn("未达标", result["3.4.2"]["status"])

    def test_http200_stream_error_and_missing_done_are_failures(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), TransportHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                config = {
                    "base_url": "http://127.0.0.1:" + str(server.server_port),
                    "timeout": 2,
                }
                body = {
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "max_tokens": 2,
                }
                for mode in ("good", "error", "partial"):
                    with patch.object(TransportHandler, "mode", mode):
                        record = call(config, body, Path(temp) / mode)
                    self.assertEqual(record["http_status"], 200)
                    self.assertEqual(
                        record["status"], "ok" if mode == "good" else "transport_error"
                    )
                    capture = Path(record["stream_capture"])
                    self.assertEqual(
                        hashlib.sha256(capture.read_bytes()).hexdigest(),
                        record["stream_sha256"],
                    )
                    with gzip.open(capture, "rt") as stream:
                        self.assertTrue(list(stream))
                    if mode == "good":
                        self.assertEqual(record["finish_reason"], "length")
                        self.assertLessEqual(
                            record["start"]["monotonic"],
                            record["first_token"]["monotonic"],
                        )
                        self.assertLessEqual(
                            record["first_token"]["monotonic"],
                            record["end"]["monotonic"],
                        )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_formal_evaluator_cannot_silently_select_one_of_full_tasks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            contract = root / "contract.json"
            contract.write_text(
                json.dumps(
                    {
                        "engine_transport": {
                            "adapter_sha256": dataset_contract.sha(
                                Path(dataset_client.__file__).with_name(
                                    "streaming_transport.py"
                                )
                            )
                        }
                    }
                )
            )
            spec = root / "spec.json"
            data = {
                "dataset": "hle",
                "plan": str(root),
                "contract_sha256": dataset_contract.sha(contract),
                "case_ids": ["case-0"],
                "diagnostic": False,
                "stage": "solver",
                "service_role": "B0",
            }
            spec.write_text(json.dumps(data))
            tasks = [{"case_id": "case-" + str(index)} for index in range(1811)]
            module = SimpleNamespace(
                load_plan=lambda _: (json.loads(contract.read_bytes()), tasks)
            )
            with patch("dataset_client.client", return_value=module):
                with self.assertRaisesRegex(ValueError, "smaller subset"):
                    dataset_client.checked(spec)
                data.update(diagnostic=True, stage="judge", service_role="B0")
                spec.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, "same pinned B1"):
                    dataset_client.checked(spec)

    def test_hle_diagnostic_keeps_both_named_subsets_and_images(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "contract.json").write_text("synthetic fixture")
            tasks = [
                {
                    "track": track,
                    "case_id": track + "-" + str(i),
                    "image": "synthetic" if i else "",
                }
                for track in ("gold", "revision")
                for i in range(3)
            ]
            module = SimpleNamespace(load_plan=lambda _: ({}, tasks))
            with patch("third_party.client", return_value=module):
                data = third_party.descriptor(
                    "hle", root, "B0", "solver", "B0", root / "spec.json", True
                )
            self.assertEqual(
                data["case_ids"], ["gold-0", "gold-1", "revision-0", "revision-1"]
            )
            self.assertTrue(data["diagnostic"])

    def test_oci_symlink_cannot_write_outside_its_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "synthetic.tar.gz"
            with tarfile.open(archive, "w:gz") as stream:
                member = tarfile.TarInfo("usr/local/escape")
                member.type = tarfile.SYMTYPE
                member.linkname = "../../../../outside"
                stream.addfile(member)
            output = root / "out"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "escaped"):
                extract_layer(archive, output.resolve())
            self.assertFalse((root / "outside").exists())

    def test_card_mapping_and_locks_follow_explicit_physical_pair(self):
        with patch.dict("os.environ", {"VLLM_HUST_PHYSICAL_DEVICES": "0,1"}):
            self.assertEqual(resource_guard.cards(), [0, 1])
            self.assertIn("/tmp/vllm-hust-engine-device-0.lock", resource_guard.paths())
            self.assertNotIn(
                "/tmp/vllm-hust-engine-device-5.lock", resource_guard.paths()
            )
            sample = SimpleNamespace(stdout="0 0 2\n1 0 3\n")
            with patch("resource_guard.subprocess.run", return_value=sample):
                resource_guard.verify_mapping("2,3")
                with self.assertRaisesRegex(ValueError, "actual selected"):
                    resource_guard.verify_mapping("0,1")
        with patch.dict("os.environ", {"VLLM_HUST_PHYSICAL_DEVICES": "0,0"}):
            with self.assertRaises(ValueError):
                resource_guard.cards()

    def test_full_output_volume_is_reported_before_sdk_or_model_work(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            args = argparse.Namespace(out=root, report_python=None, prepare_only=False)
            with (
                patch("third_party.prepare") as prepare,
                patch(
                    "third_party.shutil.disk_usage",
                    return_value=SimpleNamespace(total=100 * 2**30, free=256 * 2**20),
                ),
                patch(
                    "report_environment.report_python",
                    return_value=(Path(sys.executable), {}),
                ),
                patch(
                    "third_party.subprocess.run",
                    return_value=SimpleNamespace(returncode=0),
                ) as command,
            ):
                self.assertEqual(third_party.run(args), 2)
            prepare.assert_not_called()
            receipt = json.loads((root / "campaign.json").read_bytes())
            self.assertIn("--out volume", receipt["failure"])
            self.assertEqual(receipt["stages"], [])
            self.assertFalse(receipt["selected_execution_complete"])
            verified = verify_campaign_bindings(root)
            self.assertIn("output-volume-admission.json", verified)
            self.assertEqual(command.call_count, 1)
            self.assertIn("third_party_report.py", str(command.call_args.args[0]))

    def test_prepare_failure_keeps_failure_exit_and_never_starts_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            args = argparse.Namespace(out=root, report_python=None, prepare_only=False)
            with (
                patch(
                    "third_party.prepare",
                    side_effect=ValueError("wrong frozen version"),
                ),
                patch(
                    "report_environment.report_python",
                    return_value=(Path(sys.executable), {}),
                ),
                patch(
                    "third_party.subprocess.run",
                    return_value=SimpleNamespace(returncode=0),
                ) as command,
            ):
                result = third_party.run(args)
            self.assertEqual(result, 2)
            data = json.loads((root / "campaign.json").read_bytes())
            self.assertFalse(data["selected_execution_complete"])
            self.assertFalse(data["formal_complete"])
            self.assertIn("wrong frozen version", data["failure"])
            self.assertEqual(command.call_count, 1)
            self.assertIn("third_party_report.py", str(command.call_args.args[0]))


if __name__ == "__main__":
    unittest.main()
