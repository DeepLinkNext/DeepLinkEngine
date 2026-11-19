"""Check the report viewer through the entry shipped to evaluators."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from acceptance_dashboard import SCHEMAS, dataset_times


def seal(folder, report):
    folder.mkdir(parents=True)
    content = json.dumps(report, ensure_ascii=False).encode("utf-8")
    (folder / "summary.json").write_bytes(content)
    (folder / "checksums.json").write_text(
        json.dumps({"summary.json": hashlib.sha256(content).hexdigest()}),
        encoding="utf-8",
    )


def generate(results, output):
    return subprocess.run(
        [
            sys.executable, "-B", "-X", "utf8", str(ROOT / "one_click.py"), "dashboard",
            "--results", str(results), "--out", str(output),
        ],
        capture_output=True,
        encoding="utf-8",
        check=False,
    )


class AcceptanceDashboardTests(unittest.TestCase):
    def test_discovers_five_dataset_reports_and_keeps_campaign_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "arbitrary-results-root"
            performance = {
                "output_tokens_per_second": 12,
                "mean_ttft_ms": 10,
                "mean_tpot_ms": None,
            }
            pair = {
                "schema": SCHEMAS[0],
                "root": "/original-run/mmlu-pair",
                "arms": {
                    "B0": {
                        "groups": {"joint200": performance},
                        "receipt": {"started": 100, "ended": 150},
                    },
                    "B1": {
                        "groups": {"joint200": performance},
                        "receipt": {"started": 160, "ended": 200},
                    },
                },
            }
            seal(results / "first-run" / "mmlu-pair-report", pair)
            campaign = {
                "schema": SCHEMAS[1],
                "run": "/original-run/five",
                "receipt": {"status": "completed", "ended": 300},
                "plan": {"env_file": "/private/fixture.env"},
                "formal_targets": {
                    "3.4.4": {
                        "display_value": "-20%",
                        "numeric_threshold_met": True,
                        "accepted": False,
                        "reason": "利用率未达标 </script>",
                    }
                },
                "current_dataset_performance": {
                    "hle-solver-B0": performance,
                    "frontier-solver-B0": performance,
                },
                "current_trace_performance": {
                    name: {"arms": {"B0": performance}}
                    for name in ("swe", "terminal")
                },
            }
            report_dir = results / "second-run-report"
            seal(report_dir, campaign)
            bad = results / "tampered-report"
            seal(bad, campaign)
            (bad / "summary.json").write_bytes(b"{}")
            original_seal = (report_dir / "checksums.json").read_bytes()

            output = root / "page" / "index.html"
            done = generate(results, output)
            self.assertEqual(done.returncode, 0, done.stderr)
            page = output.read_text(encoding="utf-8")
            payload = json.JSONDecoder().raw_decode(page.split("const DATA=", 1)[1])[0]
            rows = {row["dataset"]: row for row in payload["records"]}
            self.assertEqual(set(rows), {"mmlu", "hle", "frontier", "swe", "terminal"})
            self.assertEqual(rows["mmlu"]["started"], 100)
            self.assertEqual(rows["mmlu"]["ended"], 200)
            self.assertIsNone(rows["hle"]["performance"]["B0"]["mean_tpot_ms"])
            self.assertFalse(rows["hle"]["targets"]["3.4.4"]["accepted"])
            self.assertEqual(len(payload["warnings"]), 1)
            self.assertIn("tampered-report", payload["warnings"][0]["report"])
            self.assertNotIn("利用率未达标 </script>", page)
            self.assertNotIn("fixture.env", page)
            self.assertEqual((report_dir / "checksums.json").read_bytes(), original_seal)

            denied = generate(results, report_dir / "report.html")
            self.assertNotEqual(denied.returncode, 0)
            self.assertFalse((report_dir / "report.html").exists())

    def test_regrade_retains_both_sets_of_dates(self):
        report = {
            "receipt": {
                "started": "2026-11-14T08:00:00Z",
                "ended": "2026-11-14T16:30:00+08:00",
            },
            "current_stages": [
                {"name": "hle-solver-B0", "status": "reused_verified"},
                {"name": "hle-solver-B1", "status": "reused_verified"},
                {"name": "hle-judge-B0", "status": "completed"},
            ],
        }
        original = {
            "B0": {"receipt": {"started": 100, "ended": 150}},
            "B1": {"receipt": {"started": 160, "ended": 200}},
        }
        self.assertEqual(
            dataset_times(report, "hle", original),
            {
                "started": 1794643200, "ended": 1794645000,
                "inference_started": 100, "inference_ended": 200, "regraded": True,
            },
        )
        self.assertIsNone(dataset_times({}, "terminal", {})["started"])
