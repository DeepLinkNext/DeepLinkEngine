"""Regression checks for the installed entry points, completion records and failures."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "benchmark/one-click-kit/scripts")]
sys.path.insert(0, str(ROOT / "tools"))
from test_public_run_all import reference
spec = importlib.util.spec_from_file_location("workflow", ROOT / "run_all.py")
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "model").mkdir()
        self.data = dict(model=str(self.root / "model"), physical_devices=[0, 7],
                         cache=str(self.root / "cache"), out=str(self.root / "result"))
        self.calls = []
        self.incomplete = False

    def write(self, relative, value):
        path = self.root / "result" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def execute(self, args, **kwargs):
        self.calls.append(args)
        if "install" in args:
            self.write("installation/installation.json", {"release": "full-answer-b1-20261117-r8"})
        if "run" in args:
            self.write("fixed-five/frozen-execution-plan.json", dict(
                optimized_deployment="/actual/b1", baseline_deployment="/actual/b0", logical_devices="0,1",
                datasets=["mmlu", "hle", "frontier", "swe", "terminal"]))
            self.write("fixed-five/campaign.json", dict(status="completed",
                       selected_execution_complete=not self.incomplete,
                       compute_reference=reference()))
            self.write("fixed-five-report/report.html", "fixture report")
        if "--context-tenant-only" in args:
            self.assertEqual(args[1], str(ROOT / "one_click.py"))
            self.assertNotIn("--engine-profile", args)
            self.assertEqual(args[args.index("--physical-devices") + 1], "0,7")
            self.assertEqual(args[args.index("--devices") + 1], "0,1")
            self.assertEqual(args[args.index("--baseline-deployment") + 1], "/actual/b0")
            self.write("context-tenant/execution.json", {"status": "completed"})
            self.write("context-tenant/paired-summary.json", dict(execution_complete=True,
                       comparison=[{"phase": phase} for phase in
                       ["4096-formal", "16384-formal", "32768-formal", "multi-tenant-formal"]]))
            self.write("context-tenant/paired-report/report.html", "fixture report")
        return types.SimpleNamespace(returncode=0)

    def invoke(self, execute=None):
        with patch("child_runner.run_stage", side_effect=execute or self.execute), \
             patch("one_click.verify_kit"), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = workflow.run(self.data)
        receipt = json.loads((self.root / "result/results.json").read_text())
        return result, receipt

    def test_full_flow_routes_supplement_without_package_profile_injection(self):
        result, receipt = self.invoke()
        self.assertEqual(result, 0)
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(receipt["third_party_verification"], "pending")
        self.assertEqual(len(receipt["evidence_sha256"]), 7)
        names = [row["name"] for row in receipt["stages"]]
        self.assertLess(names.index("verify-installed-tools"), names.index("run-context-tenant"))

    def test_reports_and_zero_exit_do_not_hide_incomplete_execution(self):
        self.incomplete = True
        result, receipt = self.invoke()
        self.assertEqual(result, 1)
        self.assertEqual(receipt["status"], "failed")
        self.assertIn("not show all", receipt["error"])

    def test_selected_three_dataset_success_cannot_pass_as_the_full_suite(self):
        def execute(args, **kwargs):
            result = self.execute(args, **kwargs)
            if "run" in args:
                path = self.root / "result/fixed-five/frozen-execution-plan.json"
                plan = json.loads(path.read_text())
                plan["datasets"] = ["mmlu", "hle", "frontier"]
                path.write_text(json.dumps(plan))
            return result
        result, receipt = self.invoke(execute)
        self.assertEqual(result, 1)
        self.assertIn("all five fixed datasets", receipt["error"])
        self.assertFalse(any("--context-tenant-only" in args for args in self.calls))

    def test_failed_install_never_starts_tests_or_retries(self):
        def failed(args, **kwargs):
            self.calls.append(args)
            return types.SimpleNamespace(returncode=0 if "prepare_datasets.py" in str(args[1]) else 9)
        result, receipt = self.invoke(failed)
        self.assertEqual(result, 1)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(receipt["stages"][-1]["name"], "install-package")
        self.assertEqual(receipt["stages"][-1]["returncode"], 9)

    def test_launch_failure_does_not_leave_running_stage(self):
        def failed(*args, **kwargs):
            raise OSError("cannot launch")
        result, receipt = self.invoke(failed)
        self.assertEqual(result, 1)
        self.assertEqual(receipt["stages"][0]["status"], "launch_failed")

    def test_interruption_retains_interrupted_stage(self):
        def interrupted(*args, **kwargs):
            raise KeyboardInterrupt()
        result, receipt = self.invoke(interrupted)
        self.assertEqual(result, 1)
        self.assertEqual(receipt["stages"][0]["status"], "interrupted")

    def test_existing_output_and_dangling_link_are_rejected(self):
        config = self.root / "evaluation.json"
        config.write_text(json.dumps(self.data))
        out = Path(self.data["out"])
        out.mkdir()
        with self.assertRaisesRegex(ValueError, "new output"):
            workflow.configuration(config)
        out.rmdir()
        try:
            out.symlink_to(self.root / "missing", target_is_directory=True)
        except OSError:
            return
        with self.assertRaisesRegex(ValueError, "new output"):
            workflow.configuration(config)

    def test_relative_paths_are_bound_to_configuration_file(self):
        config = self.root / "evaluation.json"
        config.write_text(json.dumps(dict(self.data, model="model", out="output", cache="cache")))
        data = workflow.configuration(config)
        self.assertEqual(data["model"], str(self.root / "model"))
        self.assertEqual(data["out"], str(self.root / "output"))

    def test_wrong_dependencies_are_rejected_without_native_imports(self):
        with patch.object(workflow.sys, "version_info", (3, 12)), \
             patch.object(workflow.platform, "python_implementation", return_value="CPython"), \
             patch.object(workflow.platform, "system", return_value="Linux"), \
             patch.object(workflow.platform, "machine", return_value="aarch64"), \
             patch.object(workflow.metadata, "version", return_value="wrong"), \
             patch("child_runner.run_stage") as execute:
            with self.assertRaisesRegex(ValueError, "expected 2.13.0"):
                workflow.preflight()
            execute.assert_not_called()

    def failed_preflight(self):
        def execute(args, **kwargs):
            if "prepare" in args:
                self.calls.append(args)
                return types.SimpleNamespace(returncode=9)
            return self.execute(args, **kwargs)
        result, _ = self.invoke(execute)
        self.assertEqual(result, 1)

    def test_bootstrap_recovery_verifies_existing_installation_and_retains_original_failure(self):
        self.failed_preflight()
        before = (self.root / "result/results.json").read_bytes()
        self.calls.clear()
        with patch("one_click.verify_kit"), patch("child_runner.run_stage", side_effect=self.execute), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = workflow.run(self.data, resume_bootstrap=True)
        self.assertEqual(result, 0)
        self.assertFalse(any("install" in args for args in self.calls))
        receipt = json.loads((self.root / "result/results.json").read_text())
        self.assertEqual(receipt["stages"][1]["mode"], "verify_existing")
        self.assertEqual(receipt["stages"][3]["mode"], "verify_existing")
        old = receipt["previous_attempts"][0]
        self.assertEqual(old["sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual((self.root / "result" / old["path"]).read_bytes(), before)
        prepare = next(args for args in self.calls if "prepare" in args)
        self.assertIn("--out", prepare)
        self.assertNotEqual(prepare[prepare.index("--out") + 1], str(self.root / "result/fixed-five-prepare"))

    def test_bootstrap_recovery_rejects_changed_physical_devices_before_work(self):
        self.failed_preflight()
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            workflow.bootstrap_resume(dict(self.data, physical_devices=[2, 3]))

    def test_bootstrap_recovery_refuses_an_existing_inference_campaign(self):
        self.failed_preflight()
        (self.root / "result/fixed-five").mkdir()
        with self.assertRaisesRegex(ValueError, "inference campaign already exists"):
            workflow.bootstrap_resume(self.data)

    def test_bootstrap_recovery_never_repeats_a_completed_workflow(self):
        self.invoke()
        with self.assertRaisesRegex(ValueError, "failed fixed-five preflight"):
            workflow.bootstrap_resume(self.data)


if __name__ == "__main__":
    unittest.main()
