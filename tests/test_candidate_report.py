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
import candidate_report
import release_selection


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
        self.assertEqual(set(summary["submission_v2"]), {"4.1", "4.2", "4.3", "4.4"})
        self.assertTrue(all(row["status"] == "not_assessed" for row in summary["submission_v2"].values()))
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
        self.assertIn("未判定", document)
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
        result = self.invoke("full_report.py", "--engine-version", "r10-native-rms", "--out", out)
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
