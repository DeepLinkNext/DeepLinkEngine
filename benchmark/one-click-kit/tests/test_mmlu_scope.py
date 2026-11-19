"""Workload identity checks, not inference or measured score fixtures."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from mmlu_scope import protocol, verify_cases


class MMLUScopeTests(unittest.TestCase):
    def test_another200_question_set_cannot_inherit_the_frozen_scope(self):
        expected = protocol()
        rows = copy.deepcopy(expected["cases"])
        rows[0]["case_id"] = "synthetic-replacement"
        with self.assertRaisesRegex(ValueError, "ordered case identities"):
            verify_cases(rows, expected)

    def test_same_questions_in_different_request_order_are_rejected(self):
        expected = protocol()
        rows = copy.deepcopy(expected["cases"])
        rows[0], rows[1] = rows[1], rows[0]
        with self.assertRaisesRegex(ValueError, "ordered case identities"):
            verify_cases(rows, expected)

    def test_prompt_change_with_unchanged_ids_is_rejected(self):
        expected = protocol()
        rows = copy.deepcopy(expected["cases"])
        rows[0]["prompt_sha256"] = "synthetic-altered-prompt"
        with self.assertRaisesRegex(ValueError, "prompt/source binding"):
            verify_cases(rows, expected)


if __name__ == "__main__":
    unittest.main()
