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
        self.source_build = json.loads((ROOT / "engine/official-build-dependencies.json").read_bytes())

    def test_all_fixed_inputs_share_the_central_release(self):
        actual = index.build_index(self.runtime, self.evaluation,
                                   dict(schema="deeplinkengine-source-build-dependency-catalog/v1", artifacts=[]))
        self.assertEqual(len(actual["artifacts"]), 14)
        combined = json.loads((ROOT / "engine/dependencies.json").read_bytes())
        combined["artifacts"] = [row for row in combined["artifacts"] if row["kind"] != "source-build"]
        self.assertEqual(actual, combined)

    def test_source_build_catalog_joins_the_index_without_changing_fixed_inputs(self):
        fixed = index.build_index(self.runtime, self.evaluation,
                                  dict(schema="deeplinkengine-source-build-dependency-catalog/v1", artifacts=[]))
        combined = index.build_index(self.runtime, self.evaluation, self.source_build)
        self.assertEqual(combined, json.loads((ROOT / "engine/dependencies.json").read_bytes()))
        self.assertEqual([row for row in combined["artifacts"] if row["kind"] != "source-build"], fixed["artifacts"])
        candidate = [row for row in combined["artifacts"] if row["kind"] == "source-build"]
        self.assertEqual(len(candidate), 1)
        self.assertEqual(candidate[0]["version"], "3.2.2")
        self.assertEqual(candidate[0]["applies_to"], ["0.28.0rc1-source"])
        self.assertEqual(next(row for row in fixed["artifacts"] if row.get("version") == "3.6.0")["sha256"],
                         "7d561749d668033e707e24be592cb1434220e81496c582e04e23a317d25ad97d")
        self.assertEqual(index.build_index(self.runtime, self.evaluation), combined)

    def test_source_build_entry_cannot_claim_an_active_fixed_scope(self):
        self.source_build["artifacts"][0]["kind"] = "python"
        with self.assertRaisesRegex(ValueError, "identity"):
            index.build_index(self.runtime, self.evaluation, self.source_build)

    def test_source_build_entry_cannot_lose_whole_hash_or_central_provenance(self):
        for key, value in (("sha256", ""), ("provenance_sha256", ""),
                           ("provenance_url", "https://example.org/receipt.json")):
            with self.subTest(key=key):
                catalog = copy.deepcopy(self.source_build)
                catalog["artifacts"][0][key] = value
                with self.assertRaises(ValueError):
                    index.build_index(self.runtime, self.evaluation, catalog)

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
