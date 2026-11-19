"""Synthetic replay fixtures; never inference or benchmark evidence."""

import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from report_execution import inspect_run


def save(root, name, value):
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value))


def seal(root):
    save(
        root,
        "checksums.json",
        {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*")
            if p.is_file() and p != root / "checksums.json"
        },
    )


def fixture(root):
    save(root, "deployment.json", {"release": "full-answer-b1-20261030-r4"})
    save(
        root,
        "receipt.json",
        {
            "kind": "continuous",
            "status": "failed",
            "quiescent": True,
            "completed": False,
            "client_started": True,
            "fixture": "synthetic",
        },
    )
    save(
        root,
        "service/runtime-identity.json",
        {"worker": "corrected_serving_runtime.CorrectedWorker"},
    )
    usage = {"prompt_tokens": 4096, "completion_tokens": 256}
    save(
        root,
        "service/good.json",
        {
            "id": "synthetic-good",
            "phase": "4096-warmup",
            "input_tokens": 4096,
            "success": True,
            "http_status": 200,
            "start": {"monotonic": 0},
            "first_token": {"monotonic": 1},
            "last_token": {"monotonic": 2},
            "end": {"monotonic": 3},
            "usage": usage,
            "finish_reason": "length",
        },
    )
    events = [
        {
            "clock": {"monotonic": 1},
            "raw": "data: "
            + json.dumps({"choices": [{"delta": {"content": "fixture"}}]}),
        },
        {
            "clock": {"monotonic": 2},
            "raw": "data: "
            + json.dumps(
                {
                    "choices": [
                        {"delta": {"content": "fixture"}, "finish_reason": "length"}
                    ],
                    "usage": usage,
                }
            ),
        },
        {"clock": {"monotonic": 3}, "raw": "data: [DONE]"},
    ]
    (root / "service/good.sse.jsonl.gz").write_bytes(
        gzip.compress(("\n".join(json.dumps(e) for e in events) + "\n").encode())
    )
    save(
        root,
        "service/bad.json",
        {
            "id": "synthetic-failed",
            "phase": "4096-warmup",
            "success": False,
            "http_status": 200,
            "start": {"monotonic": 0},
            "end": {"monotonic": 3},
            "error": "SSE error",
        },
    )
    event = {
        "clock": {"monotonic": 2},
        "raw": 'data: {"error":{"message":"synthetic failure"}}',
    }
    (root / "service/bad.sse.jsonl.gz").write_bytes(
        gzip.compress((json.dumps(event) + "\n").encode())
    )
    seal(root)


class ReportExecutionTests(unittest.TestCase):
    def test_http200_error_stream_remains_in_failed_denominator(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            data = inspect_run(root)
            self.assertEqual(
                (
                    data["recorded_requests"],
                    data["successful_requests"],
                    data["failed_requests"],
                ),
                (2, 1, 1),
            )
            self.assertFalse(data["formal_complete"])
            self.assertEqual(len(data["formal_targets"]), 4)
            self.assertTrue(
                all(
                    t["status"] == "unassessable"
                    and t["measured_value"] is None
                    and not t["accepted"]
                    for t in data["formal_targets"].values()
                )
            )

    def test_phase_throughput_uses_elapsed_wall_and_preserves_failed_denominator(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            save(
                root,
                "service/continuous.json",
                {
                    "phases": [
                        {
                            "name": "4096-warmup",
                            "start": {"monotonic": 0},
                            "end": {"monotonic": 3},
                        }
                    ]
                },
            )
            seal(root)
            group = inspect_run(root)["groups"]["4096-warmup"]
            self.assertEqual(group["recorded"], 2)
            self.assertEqual(group["failed"], 1)
            self.assertAlmostEqual(group["output_tokens_per_second"], 256 / 3)
            save(
                root,
                "service/continuous.json",
                {
                    "phases": [
                        {
                            "name": "4096-warmup",
                            "start": {"monotonic": 0},
                            "end": {"monotonic": 2},
                        }
                    ]
                },
            )
            seal(root)
            with self.assertRaisesRegex(ValueError, "Phase clock"):
                inspect_run(root)

    def test_partial_joint_rescores_raw_content_and_keeps_planned200_denominator(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            deployment = Path(temp) / "synthetic-deployment"
            fixture(root)
            for path in (root / "service").glob("bad*"):
                path.unlink()
            save(
                root,
                "receipt.json",
                {
                    "kind": "joint",
                    "status": "failed",
                    "quiescent": True,
                    "completed": False,
                    "client_started": True,
                },
            )
            save(root, "deployment.json", {"root": str(deployment), "release": "full-answer-b1-20261030-r4"})
            good = root / "service/good.json"
            row = json.loads(good.read_bytes())
            row.pop("id")
            row.pop("phase")
            row.pop("input_tokens")
            row.update(
                index=0,
                case_id="synthetic-0",
                gold="A",
                answer="A",
                correct=True,
                output="fixturefixture",
                reasoning="",
            )
            # The scorer below returns A from this synthetic wire content.
            save(root, "service/good.json", row)
            cases = [{"case_id": f"synthetic-{i}", "answer": "A"} for i in range(200)]
            save(
                deployment, "inputs/verified-mmlu-quality-inputs.json", {"cases": cases}
            )
            scorer = deployment / "scorers/mmlu_pro_pyramidkv.py"
            scorer.parent.mkdir(parents=True)
            scorer.write_text(
                "def extract_answer(text):\n    return 'A' if text == 'fixturefixture' else None\n"
            )
            scorer_hash = hashlib.sha256(scorer.read_bytes()).hexdigest()
            requests = [
                {"synthetic": i, "max_completion_tokens": 8192} for i in range(200)
            ]
            save(
                deployment,
                "inputs/mmlu-quality018-r2/quality-request-plan.json",
                {"requests": requests, "scorer_sha256": scorer_hash},
            )
            save(
                root,
                "service/request-plan.json",
                {
                    "requests": requests,
                    "scorer_sha256": scorer_hash,
                    "request_count": 200,
                    "request_rate": "inf",
                    "warmups": 0,
                },
            )
            tokens = [
                {
                    "case_id": f"synthetic-{i}",
                    "count": 4096,
                    "token_ids_sha256": "synthetic-hash",
                }
                for i in range(200)
            ]
            save(root, "service/tokenize-preflight.json", tokens)
            save(
                deployment,
                "inputs/mmlu-quality018-r2/quality-tokenize-preflight.json",
                tokens,
            )
            seal(root)
            with patch("report_execution.verify_installed_sources"):
                data = inspect_run(root)
            self.assertEqual(data["quality"]["correct"], 1)
            self.assertEqual(data["quality"]["planned_denominator"], 200)
            self.assertEqual(data["quality"]["unreturned"], 199)
            self.assertEqual(data["quality"]["accuracy_pct_of_planned200"], 0.5)
            # A matching locally re-sealed summary cannot replace frozen requests.
            requests[0]["max_completion_tokens"] = 1
            save(
                root,
                "service/request-plan.json",
                {
                    "requests": requests,
                    "scorer_sha256": scorer_hash,
                    "request_count": 200,
                    "request_rate": "inf",
                    "warmups": 0,
                },
            )
            seal(root)
            with patch("report_execution.verify_installed_sources"):
                with self.assertRaisesRegex(ValueError, "request protocol"):
                    inspect_run(root)

    def test_joint_report_reaudits_its_bound_continuous_prerequisite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            receipt = {
                "kind": "joint",
                "status": "failed",
                "quiescent": True,
                "completed": False,
                "client_started": False,
            }
            save(root, "receipt.json", receipt)
            save(root, "deployment.json", {"root": "/synthetic-deployment", "release": "full-answer-b1-20261030-r4"})
            bound = {
                "run": "/synthetic-continuous",
                "requests": 617,
                "seal_sha256": "synthetic-seal",
            }
            save(root, "continuous-qualification.json", bound)
            seal(root)
            with patch("report_execution.qualify", return_value=bound) as audit:
                data = inspect_run(root)
            audit.assert_called_once_with(
                Path(bound["run"]), Path("/synthetic-deployment")
            )
            self.assertEqual(data["continuous_qualification"], bound)
            self.assertIn("资格已复核", data["formal_targets"]["3.4.3"]["reason"])
            self.assertFalse(data["formal_complete"])
            with patch(
                "report_execution.qualify", return_value={**bound, "requests": 616}
            ):
                with self.assertRaisesRegex(ValueError, "qualification receipt"):
                    inspect_run(root)

    def test_resealed_scalar_drift_is_rejected_against_the_raw_stream(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            path = root / "service/good.json"
            row = json.loads(path.read_bytes())
            row["first_token"]["monotonic"] = 1.1
            path.write_text(json.dumps(row))
            seal(root)
            with self.assertRaisesRegex(ValueError, "stream/scalar mismatch"):
                inspect_run(root)


if __name__ == "__main__":
    unittest.main()
