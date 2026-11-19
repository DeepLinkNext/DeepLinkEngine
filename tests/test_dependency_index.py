import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dependency_index", ROOT / "tools/verify_dependency_index.py")
index = importlib.util.module_from_spec(spec)
spec.loader.exec_module(index)


class DependencyIndexTests(unittest.TestCase):
    def setUp(self):
        self.runtime = json.loads((ROOT / "engine/r8-lock.json").read_bytes())
        self.evaluation = json.loads((ROOT / "benchmark/one-click-lock.json").read_bytes())

    def test_all_twelve_fixed_inputs_share_the_central_release(self):
        actual = index.build_index(self.runtime, self.evaluation)
        self.assertEqual(len(actual["artifacts"]), 12)
        self.assertEqual(actual, json.loads((ROOT / "engine/dependencies.json").read_bytes()))

    def test_external_active_wheel_url_is_rejected(self):
        self.runtime["environment_sources"]["torch"]["url"] = "https://example.org/torch.whl"
        with self.assertRaisesRegex(ValueError, "outside"):
            index.build_index(self.runtime, self.evaluation)

    def test_missing_fixed_vendor_dependency_is_rejected(self):
        del self.runtime["environment_sources"]["torch-npu"]
        with self.assertRaisesRegex(ValueError, "missing"):
            index.build_index(self.runtime, self.evaluation)

    def test_wrong_mirrored_filename_is_rejected(self):
        self.runtime["assets"][0]["deepengine_url"] += ".wrong"
        with self.assertRaisesRegex(ValueError, "outside"):
            index.build_index(self.runtime, self.evaluation)

    def test_reference_evidence_must_use_the_same_release(self):
        self.evaluation["reference_evidence"]["url"] = self.evaluation["reference_evidence"]["upstream_url"]
        with self.assertRaisesRegex(ValueError, "outside"):
            index.build_index(self.runtime, self.evaluation)

    def test_pip_find_links_must_match_the_declared_release(self):
        self.runtime["dependency_release"]["find_links_url"] += "-wrong"
        with self.assertRaisesRegex(ValueError, "pip index"):
            index.build_index(self.runtime, self.evaluation)


if __name__ == "__main__":
    unittest.main()
