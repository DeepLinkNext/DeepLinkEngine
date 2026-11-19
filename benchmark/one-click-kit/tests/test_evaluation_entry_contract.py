"""Independent business-contract and historical-regression checks; CPU only."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "entry_acceptance", ROOT / "ci/evaluation_contract.py"
)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class EntryContractTests(unittest.TestCase):
    def test_real_source_routes_obey_independent_full_quick_policy(self):
        self.assertEqual(len(gate.verify(ROOT)), 8)

    def test_historical_three_dataset_default_is_detected(self):
        self.assert_mutation_rejected(
            '"full": "mmlu,hle,frontier,swe,terminal"',
            '"full": "mmlu,hle,frontier"',
            "expected",
        )

    def test_silently_disabling_reference_is_detected(self):
        self.assert_mutation_rejected(
            '    if subcommand == "prepare":\n        args.append("--prepare-only")',
            '    args.append("--performance-only")\n    if subcommand == "prepare":\n        args.append("--prepare-only")',
            "reference disabled",
        )

    def assert_mutation_rejected(self, before, after, reason):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            # A private copy: do not mutate the checkout or a live frozen run.
            for name in (
                "one_click.py",
                gate.POLICY,
                *gate.policy(ROOT)["frozen_policy_sha256"],
            ):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
            for folder in ("scripts",):
                shutil.copytree(
                    ROOT / folder,
                    root / folder,
                    ignore=shutil.ignore_patterns("__pycache__"),
                )
            source = root / "one_click.py"
            text = source.read_text()
            self.assertEqual(
                text.count(before), 1, "Mutation no longer hits the intended code"
            )
            source.write_text(text.replace(before, after))
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(ROOT / "ci/evaluation_contract.py"),
                    "--root",
                    str(root),
                    "--report",
                    str(Path(tmp) / "report.json"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0, result.stdout)
            report = json.loads((Path(tmp) / "report.json").read_bytes())
            self.assertFalse(report["passed"])
            self.assertIn(reason, report["error"])

    def test_frozen_sampling_and_budgets_do_not_shrink(self):
        mmlu = json.loads((ROOT / "protocols/mmlu-fixed200.json").read_bytes())
        hle = json.loads((ROOT / "protocols/hle-fixed200.json").read_bytes())
        trace = json.loads(
            (ROOT / "protocols/public-agent-trace-fixed32-v1.json").read_bytes()
        )
        self.assertEqual(mmlu["tasks_per_arm"], 200)
        self.assertEqual(mmlu["generation"]["max_completion_tokens"], 8192)
        self.assertEqual(hle["selected_tasks"], 200)
        self.assertEqual([s["selected"] for s in hle["strata"]], [64, 10, 113, 13])
        self.assertEqual(
            (
                hle["generation"]["solver_max_tokens"],
                hle["generation"]["judge_max_tokens"],
            ),
            (16384, 8192),
        )
        self.assertEqual(trace["sampling"]["tasks_per_dataset"], 32)
        self.assertEqual(
            (trace["generation"]["max_tokens"], trace["concurrency"]), (2048, 8)
        )
        for name in ("swe", "terminal"):
            self.assertEqual(
                len(trace["datasets"][name]["scope"]["selected_task_ids"]), 32
            )


if __name__ == "__main__":
    unittest.main()
