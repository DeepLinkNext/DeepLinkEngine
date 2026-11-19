import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from monitor_stage import locate


class MonitorStageTests(unittest.TestCase):
    def write(self, root, arm, status):
        p = root / "mmlu-pair" / arm / "receipt.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"status": status}))

    def test_monitor_follows_pair_from_B0_to_B1(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write(root, "B0", "running-client")
            self.assertEqual(locate(root, "mmlu-pair")[1], "mmlu-pair/B0")
            self.write(root, "B0", "completed")
            self.write(root, "B1", "starting-model")
            self.assertEqual(locate(root, "mmlu-pair")[1], "mmlu-pair/B1")

    def test_final_failure_keeps_the_failed_arm_visible(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write(root, "B0", "failed")
            final = {"stages": [{"name": "mmlu-pair", "status": "failed"}]}
            self.assertEqual(locate(root, None, final)[1], "mmlu-pair/B0")

    def test_single_stage_and_no_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertEqual(locate(root, "hle-solver-B0"), (root / "hle-solver-B0", "hle-solver-B0"))
            self.assertEqual(locate(root, None), (None, None))
            with self.assertRaises(ValueError):
                locate(root, "../other")
