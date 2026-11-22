"""CPU regression using both real, published r10 candidate summaries."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark import candidate_report, candidate_v2
from engine import release_selection


CPU_ENTRY = """
from pathlib import Path
import runpy
import sys

def reject_side_effects(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "subprocess.Popen", "os.exec", "os.system"}:
        raise AssertionError(f"CPU report attempted {event}")

sys.argv = sys.argv[1:]
sys.path.insert(0, str(Path(sys.argv[0]).parent))
sys.addaudithook(reject_side_effects)
runpy.run_path(sys.argv[0], run_name="__main__")
"""


class CandidateReportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.temp = Path(temp.name)

    def invoke(self, program, *args):
        return subprocess.run(
            [sys.executable, "-B", "-c", CPU_ENTRY, str(ROOT / program), *map(str, args)],
            env={**os.environ, "PYTHONUTF8": "1",
                 release_selection.SELECTION: str(self.temp / "unavailable-stable-selection.json")},
            capture_output=True, encoding="utf-8", timeout=30,
        )

    def check_report(self, out, alias, correct):
        summary = json.loads((out / "summary.json").read_bytes())
        receipt = json.loads((out / "receipt.json").read_bytes())
        binding = release_selection.archived_candidate(alias)
        self.assertEqual(summary["selection"], receipt["selection"])
        self.assertEqual(summary["selection"]["version"], alias)
        self.assertEqual(summary["selection"]["status"], "candidate")
        self.assertEqual(summary["selection"]["baseline_id"], "original018-cann8.5.1")
        self.assertEqual(receipt["source"]["sha256"], binding["sha256"])
        self.assertEqual(receipt["mode"], "cpu-replay")
        self.assertEqual((receipt["model_requests"], receipt["npu_tasks_started"]), (0, 0))
        self.assertEqual(len(receipt["workloads"]), 7)
        v2 = summary["submission_v2"]
        self.assertEqual(set(v2), {"4.1", "4.2", "4.3", "4.4"})
        self.assertEqual(v2["4.2"]["evidence_scope"], "cpu_audited_originals")
        self.assertTrue(v2["4.2"]["raw_reaudit"]["met"])
        self.assertFalse(v2["4.2"]["raw_missing"])
        self.assertIn("original_evidence_audit", receipt["assessment_sources"])
        fixed = alias == "r10-native-rms"
        for level in ("milestone", "final"):
            self.assertEqual(v2["4.2"][level]["status"], "met")
            self.assertIn("mmlu", v2["4.2"][level]["datasets"])
            expected_compute = (level == "milestone") if fixed else None
            self.assertIs(v2["4.1"][level]["met"], expected_compute)
            self.assertIs(v2["4.3"][level]["met"], not fixed)
        if fixed:
            compute = v2["4.1"]
            self.assertAlmostEqual(float(compute["efficiency_lower_percent"]["B1"]), 89.14006, places=5)
            self.assertEqual(compute["native"]["profile"]["max_tokens"], 256)
            self.assertEqual(compute["native"]["profile"]["formal_repetitions"], 3)
            self.assertIn("compute_audit", receipt["assessment_sources"])
        self.assertTrue(v2["4.4"]["milestone"]["met"])
        self.assertTrue(v2["4.4"]["final"]["met"])
        reviewed = v2["4.3"]
        self.assertEqual(reviewed["physical_cards"], {"B0": [1, 6], "B1": [0, 3]})
        self.assertAlmostEqual(float(reviewed["measured_ratios"]["ttft_p95"]), 3.10046 if fixed else 1.60346, places=5)
        self.assertAlmostEqual(float(reviewed["measured_window_ratios"]["ttft_p99"]), 2.34825 if fixed else 1.39701, places=5)
        self.assertIn("context_audit", receipt["assessment_sources"])
        speed = v2["4.2"]["datasets"]
        self.assertEqual(speed["hle"]["milestone"]["status"], "met")
        self.assertEqual(speed["hle"]["final"]["status"], "not_met")
        expected_track_drop = {"olympiad": "1", "research": "0"} if fixed else {"olympiad": "6", "research": "-5"}
        self.assertEqual(speed["frontier"]["quality"]["drop_pp"], expected_track_drop)
        self.assertEqual(speed["frontier"]["final"]["met"], fixed)
        self.assertEqual(speed["mmlu"]["quality"]["drop_pp"], "-2.5" if fixed else "2.5")
        self.assertEqual(speed["hle"]["quality"]["source"], "paired_grade_transitions" if fixed else "candidate_summary")
        self.assertEqual(("priority_grades" in summary["assessment_sources"]), fixed)
        self.assertEqual(receipt["assessment_sources"], summary["assessment_sources"])
        cost = v2["4.4"]
        self.assertEqual(cost["cost_basis"], "published_rental")
        self.assertEqual(cost["published_rental"]["monthly_cost_per_card_cny"], "1875")
        self.assertEqual(receipt["published_rental_sha256"], cost["published_rental"]["evidence_sha256"])
        mmlu = cost["mmlu_full_answers"]
        self.assertTrue(mmlu["cost_gate"]["met"])
        self.assertAlmostEqual(float(mmlu["unit_cost_reduction_percent"]), 66.03 if fixed else 66.01, places=2)
        self.assertEqual(mmlu["arms"]["B0"]["measurement_seal_sha256"],
                         summary["archived"]["measurements"]["mmlu"]["B0"]["source_seal_sha256"])
        self.assertAlmostEqual(mmlu["utilization"]["B0"]["mean_percent"], 33.2272, places=4)
        self.assertAlmostEqual(mmlu["utilization"]["B1"]["mean_percent"], 79.0069 if fixed else 79.8770, places=4)
        self.assertAlmostEqual(float(mmlu["same_window_utilization_gain_pp"]), 45.7797 if fixed else 46.6498, places=4)
        self.assertAlmostEqual(float(cost["four_tenant"]["arms"]["B0"]["cost_per_million_output_tokens_cny"]), 11.2142, places=4)
        self.assertAlmostEqual(float(cost["four_tenant"]["unit_cost_reduction_percent"]), 63.65056 if fixed else 66.93049, places=5)
        self.assertAlmostEqual(float(cost["four_tenant"]["ai_core_mean_percent"]), 86.35871 if fixed else 86.67085, places=5)
        self.assertEqual(summary["archived"], json.loads((ROOT / binding["path"]).read_bytes()))
        self.assertEqual(summary["archived"]["measurements"]["mmlu"]["candidate"]["correct"], correct)
        for profile, row in summary["workloads"].items():
            for role, computed in row["arms"].items():
                arm = summary["archived"]["measurements"][profile][role]
                self.assertEqual(computed["output_tokens_per_second"], arm["output_tokens"] / arm["duration_seconds"])
        for name, pin in receipt["outputs"].items():
            raw = (out / name).read_bytes()
            self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), (pin["bytes"], pin["sha256"]))
        document = (out / "report.html").read_text(encoding="utf-8")
        self.assertIn("Terminal-Bench 2.1", document)
        self.assertIn("合成闭环 4K / C8", document)
        self.assertIn("已按原件封存复算", document)
        self.assertIn("4.2 逐数据集门槛核验", document)
        self.assertIn("4.4 按公开租金折算", document)
        self.assertIn("tenant-3", document)
        return summary

    def test_run_all_explicit_candidate_on_cpu(self):
        out = self.temp / "native-rms"
        result = self.invoke("run_all.py", "--report-only", "--engine-version", "r10-native-rms", "--out", out)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.check_report(out, "r10-native-rms", 144)

    def test_one_click_direct_candidate_on_cpu(self):
        out = self.temp / "contiguous"
        release = "full-answer-b1-20261121-r10-contiguous-restore"
        result = self.invoke("one_click.py", "report", "--engine-version", release, "--out", out)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = self.check_report(out, "r10-contiguous-restore", 134)
        changes = data["workloads"]["16384-formal"]["candidate_change_vs"]
        self.assertAlmostEqual(changes["native_rms"]["throughput_percent"], 10.16, places=2)

    def test_run_all_report_config_uses_relative_output_path(self):
        config = self.temp / "r10.json"
        config.write_text(json.dumps({"engine_version": "r10-contiguous-restore", "out": "report"}), encoding="utf-8")
        result = self.invoke("run_all.py", "--config", config, "--report-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.check_report(self.temp / "report", "r10-contiguous-restore", 134)

    def test_candidate_rejected_by_execution_paths(self):
        config = self.temp / "r10.json"
        config.write_text(json.dumps({"engine_version": "r10-native-rms", "out": "never"}), encoding="utf-8")
        for program, args in (("run_all.py", ("--config", config)),
                              ("one_click.py", ("run", "--config", config))):
            result = self.invoke(program, *args)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("archived candidate", result.stderr)
        with patch.object(release_selection, "read_json", side_effect=AssertionError("network")):
            with self.assertRaisesRegex(ValueError, "archived candidate"):
                release_selection.select(self.temp, version="r10-native-rms")
        self.assertFalse((self.temp / "never").exists())

    def test_unknown_conflicting_and_existing_outputs_are_rejected(self):
        out = self.temp / "report"
        for version, flags in (("r10", ()), ("r10-native-rms", ("--prepare-environment",)),
                               ("r10-native-rms", ("--rerun-b0",))):
            result = self.invoke("run_all.py", "--report-only", "--engine-version", version, "--out", out, *flags)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertFalse(out.exists())
        out.mkdir()
        marker = out / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        result = self.invoke("benchmark/full_report.py", "--engine-version", "r10-native-rms", "--out", out)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
        self.assertEqual(list(out.iterdir()), [marker])

    def test_unknown_configured_candidate_does_not_replay_an_existing_run(self):
        config = self.temp / "unknown.json"
        config.write_text(json.dumps({"engine_version": "r10-typo", "out": "report", "cache": "cache"}),
                          encoding="utf-8")
        result = self.invoke("run_all.py", "--config", config, "--report-only")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("Unrecognized report engine version", result.stderr)
        self.assertFalse((self.temp / "report").exists())

    def test_missing_candidate_track_grade_stays_unknown(self):
        _, archive = candidate_report.load_archive("r10-contiguous-restore")
        del archive["measurements"]["frontier"]["candidate"]["track_scores"]
        quality, proof = candidate_v2.grade_details(archive, {})
        result = candidate_v2.assess_speed(archive, quality, proof)
        row = result["datasets"]["frontier"]
        self.assertIsNone(row["quality"]["met"])
        self.assertIn("缺项运行臂：B1", row["quality"]["reason"])
        self.assertEqual(row["final"]["status"], "missing_evidence")
        self.assertTrue(result["final"]["met"])

    def test_priority_grades_reject_wrong_run(self):
        binding, archive = candidate_report.load_archive("r10-native-rms")
        priority = json.loads((ROOT / binding["quality_source"]).read_bytes())
        priority["priority_pair"]["scope"]["frontier"]["r10_corrected"]["source_seal_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "different solver"):
            candidate_v2.recover_priority_tracks(archive, priority)

    def test_fixed_b0_sampling_rejects_mismatched_window(self):
        evidence = json.loads((ROOT / candidate_report.FIXED_B0_V2).read_bytes())["mmlu"]
        meter = {**evidence["meter"], "window_seconds": "1"}
        self.assertIsNone(candidate_v2.matched_utilization(evidence, meter))

    def test_original_evidence_cannot_cross_candidate_boundaries(self):
        _, archive = candidate_report.load_archive("r10-contiguous-restore")
        document = json.loads((ROOT / candidate_report.ORIGINAL_AUDIT).read_bytes())
        original = document["candidates"]["r10-native-rms"]
        _, proof = candidate_v2.grade_details(archive, {})
        with self.assertRaisesRegex(ValueError, "another candidate"):
            candidate_v2.validate_original(archive, original, proof)

    def test_changed_archived_bytes_are_rejected_before_output(self):
        record = release_selection.archived_candidate("r10-native-rms")
        source = self.temp / record["path"]
        source.parent.mkdir(parents=True)
        source.write_bytes((ROOT / record["path"]).read_bytes() + b" ")
        out = self.temp / "out"
        with patch.object(candidate_report, "ROOT", self.temp):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                candidate_report.render("r10-native-rms", out)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
