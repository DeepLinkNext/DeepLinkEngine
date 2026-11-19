"""CPU orchestration regressions; these fixtures do not claim NPU execution."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmark/one-click-kit/scripts"))
import run_all


def reference():
    arm = dict(F_interval=[100, 110], T_seconds="2",
        achieved_tflops_per_card_interval=["0.000000000025", "0.0000000000275"],
        reference_tflops_per_card_upper="0.00000000005", efficiency_lower_percent="50")
    return dict(status="measured", certificate_kind="qualified_full_reference_denominator_reuse/v1",
        full_model_certificate=True, full_reference_denominator_qualified=True,
        original_reference_work_and_time_retained=True, original_efficiency_used=False,
        shared_work_cancellation_used=False, arms=dict(B0=arm, B1=arm), milestone_met=True)


class FullSuiteEntryTests(unittest.TestCase):
    def execute(self, temp, *, retired=False, failure=False, interrupt=False):
        out = Path(temp) / "new-run"
        commands = []

        def command(args):
            commands.append(args)
            if "install" in args:
                record = out / "installation/installation.json"
                record.parent.mkdir(parents=True, exist_ok=True)
                record.write_text(json.dumps(dict(release="full-answer-b1-20261117-r8")))
            if "run" in args and str(ROOT / "one_click.py") in args:
                if interrupt:
                    raise KeyboardInterrupt("cancelled fixture")
                if failure:
                    return SimpleNamespace(returncode=2)
                fixed = out / "fixed-five"
                fixed.mkdir()
                current = reference()
                if retired:
                    current["certificate_kind"] = "matched_complete_necessary_work_and_full_reference_lower_bound"
                (fixed / "campaign.json").write_text(json.dumps(dict(
                    status="completed", selected_execution_complete=True, compute_reference=current)))
                (fixed / "frozen-execution-plan.json").write_text(json.dumps(dict(
                    optimized_deployment="/new/sdk", baseline_deployment="/new/b0", logical_devices="0,1")))
                report = out / "fixed-five-report/report.html"
                report.parent.mkdir()
                report.write_text("fixture only")
            if "--context-tenant-only" in args:
                folder = out / "context-tenant"
                folder.mkdir()
                (folder / "execution.json").write_text(json.dumps(dict(status="completed")))
                (folder / "paired-summary.json").write_text(json.dumps(dict(execution_complete=True,
                    comparison=[dict(phase=p) for p in ("4096-formal", "16384-formal", "32768-formal", "multi-tenant-formal")])))
                report = out / "context-tenant/paired-report/report.html"
                report.parent.mkdir(parents=True)
                report.write_text("fixture only")
            return SimpleNamespace(returncode=0)

        with patch("one_click.verify_kit"), patch("child_runner.run_stage", side_effect=command):
            status = run_all.run(dict(out=str(out), model="/model", physical_devices=[1, 6], cache="/cache"))
        return status, json.loads((out / "results.json").read_text()), commands, out

    def test_public_suite_runs_corrected_entry_and_automatic_reference_before_supplement(self):
        with tempfile.TemporaryDirectory() as temp:
            status, receipt, commands, out = self.execute(temp)
            self.assertEqual(status, 0)
            self.assertEqual(receipt["status"], "completed")
            self.assertEqual(len(commands), 8)
            self.assertTrue(all("evaluate" not in args for args in commands))
            entry_calls = [args for args in commands if str(ROOT / "one_click.py") in args]
            self.assertEqual(len(entry_calls), 3)
            self.assertTrue(all(args[:2] == [sys.executable, str(ROOT / "one_click.py")] for args in entry_calls))
            config = json.loads((out / "evaluation.json").read_text())
            self.assertEqual(config["reference_profile"], "submission-v2")
            self.assertTrue(receipt["compute_reference_target"]["accepted"])
            self.assertEqual(receipt["third_party_verification"], "pending")

    def test_zero_exit_with_retired_partial_reference_still_fails_before_supplement(self):
        with tempfile.TemporaryDirectory() as temp:
            status, receipt, commands, _ = self.execute(temp, retired=True)
            self.assertEqual(status, 1)
            self.assertEqual(receipt["status"], "failed")
            self.assertIn("reference remains incomplete", receipt["error"])
            self.assertFalse(any("--context-tenant-only" in args for args in commands))

    def test_failed_child_stops_pipeline_and_retains_returncode(self):
        with tempfile.TemporaryDirectory() as temp:
            status, receipt, commands, _ = self.execute(temp, failure=True)
            self.assertEqual(status, 1)
            self.assertEqual(receipt["stages"][-1]["returncode"], 2)
            self.assertEqual(len(commands), 6)

    def test_cancelled_child_is_recorded_as_failed_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            status, receipt, commands, _ = self.execute(temp, interrupt=True)
            self.assertEqual(status, 1)
            self.assertEqual(receipt["stages"][-1]["status"], "interrupted")
            self.assertIn("cancelled fixture", receipt["stages"][-1]["error"])


if __name__ == "__main__":
    unittest.main()
