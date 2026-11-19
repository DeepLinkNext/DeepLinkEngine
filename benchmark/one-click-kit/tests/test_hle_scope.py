"""Scope integrity regressions without model calls or synthetic quality claims."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import hle_scope


def fixture():
    tasks = []
    for (track, modality), n in zip(
        hle_scope.STRATA, hle_scope.POPULATION, strict=True
    ):
        for i in range(n):
            tasks.append(
                {
                    "case_id": f"{track}-{modality}-{i}",
                    "track": track,
                    "row_index": len(tasks),
                    "image": "synthetic-image" if modality == "image" else "",
                    "problem": "synthetic original question",
                    "answer": "synthetic answer",
                }
            )
    return tasks


class HLEScopeTests(unittest.TestCase):
    def test_proportional_scope_has_four_strata_and_no_replacement(self):
        tasks = fixture()
        sample = hle_scope.select(tasks)
        self.assertEqual(hle_scope.allocate(), [64, 10, 113, 13])
        self.assertEqual(len(sample), 200)
        self.assertEqual(len({x["case_id"] for x in sample}), 200)
        self.assertEqual(sum(bool(x["image"]) for x in sample), 23)
        self.assertNotEqual(sample, tasks[:200])
        self.assertEqual(sample, hle_scope.select(tasks))

    def test_ids_order_and_original_content_are_bound_not_just_count(self):
        sample = hle_scope.select(fixture())
        expected = hle_scope.entries(sample)
        hle_scope.verify_entries(sample, expected)
        mutations = [
            list(reversed(sample)),
            copy.deepcopy(sample),
            copy.deepcopy(sample),
        ]
        mutations[1][0]["problem"] += " changed"
        mutations[2][0]["image"] = "changed modality"
        for changed in mutations:
            with self.assertRaisesRegex(ValueError, "IDs/order/original task bytes"):
                hle_scope.verify_entries(changed, expected)

    def test_preparation_only_changes_scope_preserving_generation_and_scoring(self):
        tasks = fixture()
        sample = hle_scope.select(tasks)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "tasks.json").write_text(json.dumps(tasks))
            protocol = root / "published.json"
            protocol.write_text(
                json.dumps(
                    {
                        "cases": hle_scope.entries(sample),
                        "strata": [],
                        "selection_method": "synthetic test",
                    }
                )
            )
            pin = hashlib.sha256(protocol.read_bytes()).hexdigest()
            contract = {
                "selection": {},
                "solver": {"max_tokens": 16384},
                "judge": {"max_tokens": 8192},
                "scoring_policy": "strict_json",
            }
            before = copy.deepcopy(contract)
            with (
                patch.object(hle_scope, "PROTOCOL", protocol),
                patch.object(hle_scope, "PROTOCOL_SHA256", pin),
            ):
                scope = hle_scope.apply(root, contract, "fixed200-v1")
                selected = json.loads((root / "tasks.json").read_bytes())
                hle_scope.verify_selected(root, contract, selected)
                (root / "sample-scope.json").write_text("changed")
                with self.assertRaisesRegex(ValueError, "Archived HLE"):
                    hle_scope.verify_selected(root, contract, selected)
            self.assertEqual(scope["selected_tasks"], 200)
            self.assertEqual(
                len(json.loads((root / "excluded.json").read_bytes())), 1611
            )
            for key in ("solver", "judge", "scoring_policy"):
                self.assertEqual(contract[key], before[key])


if __name__ == "__main__":
    unittest.main()
