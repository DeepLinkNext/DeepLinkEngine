"""Check that a failed prerequisite cannot submit a joint workload."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from reproduce_b1_r4 import reproduce


class CampaignTests(unittest.TestCase):
    def test_failed_prerequisite_stops_without_retry_or_joint_submission(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "campaign"
            args = argparse.Namespace(
                out=root,
                deployment=Path("/synthetic"),
                devices="2,3",
                model=Path("/synthetic-model"),
                report_python=None,
            )
            with patch(
                "reproduce_b1_r4.subprocess.run",
                return_value=subprocess.CompletedProcess([], 2),
            ) as run:
                self.assertEqual(reproduce(args), 2)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0][-1], "continuous")
            receipt = json.loads((root / "campaign.json").read_bytes())
            self.assertEqual(receipt["unexecuted"], ["joint"])
            self.assertFalse(receipt["completed"])
            self.assertFalse(receipt["formal_complete"])


if __name__ == "__main__":
    unittest.main()
