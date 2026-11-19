"""Synthetic integrity and scope regressions, never benchmark results."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import project_evidence
from report_execution import inspect_run
from test_report_execution import fixture
from telemetry_audit import utilization


def raw(a, b):
    return f"| 5 910B2 | OK | 10 |\n| 0 | 0000:02:00.0 | {a} |\n| 6 910B2 | OK | 10 |\n| 0 | 0000:41:00.0 | {b} |"


class ProjectEvidenceTests(unittest.TestCase):
    def test_local_manifest_rewrite_cannot_bless_modified_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "frozen.txt"
            source.write_bytes(b"original")
            data = {
                "commit": project_evidence.BENCHMARK_COMMIT,
                "files": {
                    "frozen.txt": {
                        "bytes": 8,
                        "sha256": project_evidence.digest(source),
                    }
                },
            }
            manifest = root / "source-readback.json"
            manifest.write_text(json.dumps(data))
            with patch.object(
                project_evidence, "MANIFEST_SHA256", project_evidence.digest(manifest)
            ):
                project_evidence.verify(root)
                source.write_bytes(b"modified")
                with self.assertRaisesRegex(ValueError, "Formal evidence changed"):
                    project_evidence.verify(root)
                data["files"]["frozen.txt"]["sha256"] = project_evidence.digest(source)
                manifest.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, "pinned benchmark snapshot"):
                    project_evidence.verify(root)

    def test_existing_evidence_does_not_turn_failed_current_run_into_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            existing = {
                "formal_targets": {"3.4.1": {"accepted": True, "measured_value": 60}}
            }
            data = inspect_run(root, existing)
            self.assertEqual(data["formal_targets"], existing["formal_targets"])
            self.assertEqual(data["receipt"]["status"], "failed")
            self.assertEqual(
                (data["successful_requests"], data["failed_requests"]), (1, 1)
            )
            self.assertFalse(data["formal_complete"])
            self.assertTrue(
                all(
                    x["measured_value"] is None
                    for x in data["current_run_targets"].values()
                )
            )

    def test_sampler_midpoints_clip_to_request_interval(self):
        rows = [
            {"monotonic_start": 0, "monotonic_end": 2, "raw": raw(10, 20)},
            {"monotonic_start": 4, "monotonic_end": 6, "raw": raw(90, 100)},
            {"monotonic_start": 10, "monotonic_end": 12, "raw": raw(0, 0)},
        ]
        result = utilization(rows, 2, 8)
        self.assertEqual(result["per_card_percent"], {"5": 50, "6": 60})
        self.assertEqual(result["mean_percent"], 55)
        self.assertEqual(result["missing_seconds"], 0)
        self.assertEqual(result["covered_seconds"], 6)
        result = utilization(rows, 0, 8)
        self.assertEqual(result["missing_seconds"], 1)
        with self.assertRaises(ValueError):
            utilization([dict(rows[0], returncode=1)], 2, 8)


if __name__ == "__main__":
    unittest.main()
