"""Lifecycle and data integrity at the multi-turn experiment boundary."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark.engine_pool import placements, running_pool
from benchmark.session_workload import run


class SessionWorkloadTest(unittest.TestCase):
    def test_existing_pool_checks_model_without_launching_or_stopping(self):
        spec = dict(instances=[dict(host="a", physical_cards=[0, 1], lifecycle="existing",
                                    endpoint="http://hot")])
        nodes = placements({}, spec, 1, 2)
        with tempfile.TemporaryDirectory() as path, patch("benchmark.engine_pool.serve") as launch:
            with patch("benchmark.engine_pool.request", return_value={"data": [{"id": "35b"}]}):
                with running_pool(nodes, Path(path), "35b"):
                    self.assertTrue((Path(path) / "model-readiness.json").exists())
                with self.assertRaisesRegex(ValueError, "model"):
                    with running_pool(nodes, Path(path), "14b"):
                        self.fail("wrong model accepted")
            launch.assert_not_called()

    def test_changed_workload_is_rejected_before_starting_engines(self):
        with tempfile.TemporaryDirectory() as path:
            csv = Path(path) / "sessions.csv"
            csv.write_text("session_id,round_idx\na,0\n")
            out = Path(path) / "result"
            with self.assertRaisesRegex(ValueError, "digest"):
                run({"request_sha256": "bad"}, {"requests": str(csv)}, out, {}, [], None)
            self.assertFalse(out.exists())
