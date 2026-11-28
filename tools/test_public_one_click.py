"""CPU regressions for the public entry and reference acceptance boundaries."""

from contextlib import redirect_stderr, redirect_stdout
from decimal import Decimal
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("public_entry", ROOT / "benchmark/one_click.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
sys.path.insert(0, str(entry.KIT / "scripts"))
from full_reference_reuse import arm_statistics, qualify_mapping
from compute_reference import target
from benchmark import run_status


class PublicEntryTests(unittest.TestCase):
    def setUp(self):
        selected=patch('engine.release_selection.select',return_value=None)
        selected.start()
        self.addCleanup(selected.stop)

    def test_reference_accepts_selected_pair_on_eight_card_host(self):
        qualify_mapping(dict(physical_cards=[2, 7], logical_cards=[2, 7],
            npu_smi_mapping="\n".join(f"{card} 0 {card} Ascend 910B2" for card in range(8))), "910B2")

    def test_reference_rejects_misidentified_selected_card(self):
        with self.assertRaisesRegex(ValueError, "mapping differs"):
            qualify_mapping(dict(physical_cards=[2, 7], logical_cards=[2, 7],
                npu_smi_mapping="2 0 2 Ascend 910B2\n7 0 7 Ascend 910B3"), "910B2")

    def test_real_public_supplement_route_reaches_frozen_parser_without_profile_injection(self):
        observed = []
        def execute(executable, command, environment):
            self.assertEqual(Path(command[1]), entry.KIT / "scripts/acceptance_suite.py")
            self.assertNotIn("--engine-profile", command)
            spec = importlib.util.spec_from_file_location("public_supplement_contract", command[1])
            suite = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(suite)
            with patch.object(sys, "argv", command[1:]), patch.object(suite, "run", return_value=0) as run:
                self.assertEqual(suite.main(), 0)
                observed.append(run.call_args.args[0])
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(entry, "reference_evidence", return_value=Path(temp)), \
                patch.object(entry, "prepare_data", side_effect=lambda args: args), \
                patch.object(entry, "prepare_baseline", side_effect=lambda args: args), \
                patch.object(entry.os, "execve", side_effect=execute), \
                patch.object(sys, "argv", list(sys.argv)):
            with self.assertRaises(SystemExit) as exited:
                entry.main(["--context-tenant-only", "--rerun-b0", "--no-follow", "--deployment", "/sdk",
                    "--baseline-deployment", "/b0", "--b1-python", sys.executable,
                    "--model", "/model", "--cache", temp, "--out", str(Path(temp) / "new-run"),
                    "--physical-devices", "1,6", "--devices", "0,1"])
            self.assertIn(exited.exception.code, (None, 0))
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0].physical_devices, "1,6")
        self.assertEqual(observed[0].devices, "0,1")

    def test_published_sources_match_every_inventory_pin(self):
        self.assertEqual(entry.verify_kit()["tools"]["files"], 510)

    def test_sealed_9b_route_uses_the_public_entry_and_frozen_base(self):
        output = io.StringIO()
        with patch.object(sys, "path", list(sys.path)), redirect_stdout(output):
            with self.assertRaises(SystemExit) as result:
                entry.main(["reference-9b", "--help"])
        self.assertEqual(result.exception.code, 0)
        self.assertIn("--conservative-reference-evidence", output.getvalue())
        self.assertEqual(entry.verify_kit(legacy=True)["tools"]["files"], 510)

    def test_changed_source_and_unlisted_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "one_click.py").write_text("original")
            manifest = dict(commit="a" * 40, files={"one_click.py": entry.digest(root / "one_click.py")})
            (root / "DEPLOYMENT-TOOLS.json").write_text(json.dumps(manifest))
            lock = root.parent / (root.name + "-lock.json")
            lock.write_text(json.dumps(dict(tools=dict(upstream_commit="a" * 40,
                manifest_sha256=entry.digest(root / "DEPLOYMENT-TOOLS.json")))))
            try:
                with patch.object(entry, "KIT", root), patch.object(entry, "LOCK", lock):
                    entry.verify_kit()
                    (root / "one_click.py").write_text("changed")
                    with self.assertRaisesRegex(ValueError, "source changed"):
                        entry.verify_kit()
                    (root / "one_click.py").write_text("original")
                    (root / "unexpected.py").write_text("added")
                    with self.assertRaisesRegex(ValueError, "inventory differs"):
                        entry.verify_kit()
            finally:
                lock.unlink()

    def test_mismatched_reference_download_fails_before_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            lock = entry.verify_kit()
            with patch.dict(os.environ, {"XDG_CACHE_HOME": temp}, clear=True), \
                    patch.object(entry.urllib.request, "urlopen", return_value=io.BytesIO(b"wrong bytes")):
                with self.assertRaisesRegex(ValueError, "size/SHA-256 mismatch"):
                    entry.reference_evidence(lock)
                self.assertFalse((Path(temp) / "vllm-hust-engine/project-evidence-53541118/benchmark-evidence").exists())

    def test_unsafe_reference_archive_never_escapes_cache(self):
        # A compromised mirror must not gain extraction privileges even if a
        # future operator updates the outer archive pin.
        body = io.BytesIO()
        with tarfile.open(fileobj=body, mode="w:gz") as archive:
            info = tarfile.TarInfo("../escape")
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
        raw = body.getvalue()
        lock = entry.verify_kit()
        lock["reference_evidence"].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        module = type("Contract", (), dict(ASSET_BYTES=len(raw), ASSET_SHA256=hashlib.sha256(raw).hexdigest()))
        with tempfile.TemporaryDirectory() as temp, \
                patch.dict(os.environ, {"XDG_CACHE_HOME": temp}, clear=True), \
                patch.object(entry, "evidence_module", return_value=module), \
                patch.object(entry.urllib.request, "urlopen", return_value=io.BytesIO(raw)):
            with self.assertRaisesRegex(ValueError, "Unsafe reference archive"):
                entry.reference_evidence(lock)

    def test_anonymous_endpoint_failure_uses_verified_public_fallback(self):
        body = io.BytesIO()
        with tarfile.open(fileobj=body, mode="w:gz") as archive:
            info = tarfile.TarInfo("benchmark-evidence/sentinel")
            info.size = 6
            archive.addfile(info, io.BytesIO(b"golden"))
        raw = body.getvalue()
        lock = entry.verify_kit()
        pin = hashlib.sha256(raw).hexdigest()
        lock["reference_evidence"].update(bytes=len(raw), sha256=pin,
            api_url="https://api.github.com/public-fixture", url="https://github.com/public-fixture")
        def verify(path):
            self.assertEqual((path / "sentinel").read_bytes(), b"golden")
        module = SimpleNamespace(ASSET_BYTES=len(raw), ASSET_SHA256=pin, verify=verify)
        with tempfile.TemporaryDirectory() as temp, \
                patch.dict(os.environ, {"XDG_CACHE_HOME": temp}, clear=True), \
                patch.object(entry, "evidence_module", return_value=module), \
                patch.object(entry.urllib.request, "urlopen", side_effect=[
                    entry.urllib.error.URLError("transient fixture"), io.BytesIO(raw)]) as opened:
            result = entry.reference_evidence(lock)
            self.assertEqual((result / "sentinel").read_bytes(), b"golden")
            self.assertEqual(opened.call_count, 2)
            for call in opened.call_args_list:
                self.assertFalse(call.args[0].has_header("Authorization"))

    def test_reference_different_work_does_not_cancel(self):
        result = arm_statistics([dict(repetition=1, F_interval=[100, 110], T_seconds="2")],
            dict(effective_flops_interval=[150, 200], sum_reference_seconds_lower="2",
                 reference_tflops_per_card_upper="0.00000000005", reference_repetitions=[1]))
        self.assertEqual(Decimal(result["efficiency_lower_percent"]), Decimal("50"))
        self.assertFalse(result["shared_work_cancellation_used"])

    def test_all_predeclared_repetitions_contribute_to_efficiency(self):
        rows = [dict(repetition=index, F_interval=[100, 100], T_seconds=str(seconds))
                for index, seconds in enumerate([1, 1, 4], 1)]
        result = arm_statistics(rows, dict(effective_flops_interval=[300, 300],
            sum_reference_seconds_lower="3", reference_tflops_per_card_upper="0.00000000005",
            reference_repetitions=[1, 2, 3]))
        self.assertEqual(result["T_seconds"], "6")
        self.assertEqual(Decimal(result["efficiency_lower_percent"]), Decimal("50"))

    def test_retired_partial_certificate_cannot_revive_old_pass(self):
        for kind in ("matched_complete_necessary_work_and_full_reference_lower_bound", "partial_dense_projection_diagnostic"):
            result = target(dict(status="measured", milestone_met=True,
                certificate_kind=kind, efficiency_lower_observed_percent=99))
            self.assertFalse(result["accepted"])
            self.assertIsNone(result["measured_value"])

    def test_unqualified_new_certificate_cannot_pass(self):
        result = target(dict(status="measured", milestone_met=True,
            certificate_kind="qualified_full_reference_denominator_reuse/v1"))
        self.assertFalse(result["accepted"])
        self.assertIsNone(result["measured_value"])


class RunStatusTests(unittest.TestCase):
    STAGES = (
        "continuous-B0", "continuous-B1", "mmlu-pair",
        "frontier-solver-B0", "frontier-solver-B1", "frontier-judge-B0", "frontier-judge-B1",
        "hle-solver-B0", "hle-solver-B1", "hle-judge-B0", "hle-judge-B1",
        "swe-trace-B0", "swe-trace-B1", "terminal-trace-B0", "terminal-trace-B1",
    )

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.run = Path(temp.name) / "run"
        self.logs = self.run.with_name(self.run.name + "-logs")

    def write(self, name, data):
        path = self.run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def plan(self, statuses, names=STAGES):
        self.write("frozen-stages.json", [dict(name=name) for name in names])
        self.write("execution-checklist.json", dict(status="preparing", stages=[
            dict(name=name, status=statuses.get(name, "planned")) for name in names
        ]))

    def invoke(self, *args):
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors), \
                patch.object(entry, "prepare_data", side_effect=AssertionError("status started preparation")):
            code = entry.main(["status", "--run", str(self.run), *args])
        return code, output.getvalue(), errors.getvalue()

    def test_historical_failure_lists_all_fifteen_stages_in_plan_order(self):
        self.plan({"continuous-B0": "completed", "continuous-B1": "failed"})
        self.write("continuous-B1/receipt.json", dict(status="failed", failure="Missing numba"))
        self.write("campaign.json", dict(status="failed", failure="Stage failed"))
        self.logs.mkdir()
        (self.logs / "exit.json").write_text('{"returncode": 2}', encoding="utf-8")
        files = {path: path.read_bytes() for path in self.run.rglob("*.json")}

        code, output, errors = self.invoke("--json")

        self.assertEqual((code, errors), (0, ""))
        value = json.loads(output)
        self.assertEqual((value["state"], value["stages_done"], value["stages_total"]), ("failed", 1, 15))
        self.assertEqual((value["stage"], value["returncode"], value["failure"]), ("continuous-B1", 2, "Missing numba"))
        self.assertEqual([row["name"] for row in value["stages"]], list(self.STAGES))
        self.assertEqual([row["status"] for row in value["stages"]], ["completed", "failed"] + ["not_run"] * 13)
        self.assertEqual(value["stages"][1]["failure"], "Missing numba")
        self.assertEqual({path: path.read_bytes() for path in self.run.rglob("*.json")}, files)

    def test_mmlu_reports_active_arm_and_request_count_with_reused_stage(self):
        self.plan({"continuous-B0": "reused_verified", "continuous-B1": "completed", "mmlu-pair": "running"})
        self.write("mmlu-pair/B0/receipt.json", dict(status="completed"))
        self.write("mmlu-pair/B1/receipt.json", dict(status="running-client"))
        self.write("mmlu-pair/B1/service/case-001.json", {})

        code, output, errors = self.invoke("--json")

        self.assertEqual((code, errors), (0, ""))
        value = json.loads(output)
        self.assertEqual((value["stage"], value["phase"], value["responses"], value["planned"]),
                         ("mmlu-pair/B1", "running-client", 1, 200))
        self.assertEqual((value["state"], value["stages_done"], value["stages"][2]["active_arm"]), ("running", 2, "B1"))

    def test_reference_progress_keeps_fixed_five_stage_count(self):
        self.plan(dict.fromkeys(self.STAGES, "completed"))
        self.write("compute-reference/automatic-plan.json", dict(stages=[
            dict(id="gdn-decode-independent-reference"), dict(id="current-native-synchronized-timing")
        ]))
        self.write("compute-reference/status.json", dict(status="running", stages=[
            dict(id="gdn-decode-independent-reference", status="completed"),
            dict(id="current-native-synchronized-timing", status="running"),
        ]))

        code, output, errors = self.invoke("--json")

        self.assertEqual((code, errors), (0, ""))
        value = json.loads(output)
        self.assertEqual((value["stages_done"], value["stages_total"]), (15, 15))
        self.assertEqual((value["reference"]["stages_done"], value["reference"]["stages_total"]), (1, 2))
        self.assertEqual(value["reference"]["stage"], "compute-reference/current-native-synchronized-timing")

    def test_watch_handles_partial_records_and_stops_at_final_snapshot(self):
        self.logs.mkdir()
        names = ("continuous-B0", "continuous-B1")

        def warming():
            self.plan({"continuous-B0": "running"}, names)
            self.write("continuous-B0/receipt.json", dict(status="warming-service"))

        def partial():
            (self.run / "execution-checklist.json").write_text('{"stages": [', encoding="utf-8")

        def running():
            self.plan({"continuous-B0": "running"}, names)
            self.write("continuous-B0/receipt.json", dict(status="running-client"))

        def complete():
            self.plan(dict.fromkeys(names, "completed"), names)
            self.write("continuous-B1/receipt.json", dict(status="completed"))
            self.write("campaign.json", dict(status="completed"))
            (self.logs / "exit.json").write_text('{"returncode": 0}', encoding="utf-8")

        actions = iter((warming, partial, running, complete))
        with patch.object(run_status.time, "sleep", side_effect=lambda _: next(actions)()):
            code, output, errors = self.invoke("--watch")

        self.assertEqual((code, errors), (0, ""))
        self.assertLess(output.index("准备中"), output.index("warming-service"))
        self.assertLess(output.index("warming-service"), output.index("running-client"))
        self.assertIn("阶段 2/2", output)
        self.assertIn("退出码 0", output)
        _, once, _ = self.invoke()
        self.assertIn(once.splitlines()[0], output)

    def test_watch_interrupt_does_not_change_run(self):
        self.plan({"continuous-B0": "running"}, ("continuous-B0",))
        path = self.run / "execution-checklist.json"
        before = path.read_bytes()
        with patch.object(run_status.time, "sleep", side_effect=KeyboardInterrupt):
            code, _, errors = self.invoke("--watch")
        self.assertEqual((code, errors), (130, ""))
        self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
