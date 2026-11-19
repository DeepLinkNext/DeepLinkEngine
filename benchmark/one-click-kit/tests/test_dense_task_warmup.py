"""Dense synthetic warmup adds task coverage without changing completed C8."""

import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import serving_warmup as warmup  # noqa: E402 - bundled scripts use a private source path


class DenseTaskWarmupTests(unittest.TestCase):
    def test_continuous_protocol_and_rows_remain_original_v2(self):
        self.assertEqual(warmup.protocol()["policy"], "observed-torch-triton-sealed-warmup-v2")
        self.assertNotIn("dense_decode_concurrencies", warmup.protocol())
        inputs = {str(n): [{"case": f"{n}-{i}"} for i in range(8)] for n in (4096, 16384)}
        cohorts = list(warmup.warmup_cohorts(inputs))
        self.assertEqual([len(c) for c in cohorts], [1, 2, 4, 8, 16, 24, 16])
        self.assertEqual(sum(map(len, cohorts)), 71)
        self.assertEqual(warmup.request_count(), 71)
        self.assertEqual(warmup.protocol()["output_tokens"], 32)

    def test_task_cohorts_cover_dense_batches_separately_from_formal_cases(self):
        inputs = {str(n): [{"case": f"{n}-{i}"} for i in range(8)] for n in (4096, 16384)}
        dense = {"request": {"messages": [{"role": "user", "content": warmup.DENSE_PROMPT}]}}
        cohorts = list(warmup.warmup_cohorts(inputs, dense))
        self.assertEqual([len(c) for c in cohorts[7:]], [1, 2, 4, 8, 16, 24])
        self.assertTrue(all(case is dense for c in cohorts[7:] for case in c))
        self.assertEqual(sum(map(len, cohorts)), warmup.request_count("joint"))
        self.assertEqual(warmup.request_count("dataset"), 126)
        policy = warmup.protocol("joint")
        self.assertEqual(policy["policy"], warmup.TASK_POLICY)
        self.assertLessEqual(24 * policy["dense_prompt_maximum_tokens"], 4096)
        self.assertTrue(policy["formal_clock_excludes_warmup"])
        self.assertTrue(policy["formal_requests_unchanged"])

    def test_old_c8_reuse_exception_stays_bound_to_its_historical_source(self):
        compat = json.loads((ROOT / "protocols/task-warmup-v3-c8-compatibility.json").read_bytes())
        self.assertEqual(compat["current_sha256"], "eeec13dc35158dcf01330e9dd80757f03a63c6ba1dbf37216b33f3718d1dc907")
        self.assertNotEqual(hashlib.sha256((ROOT / "scripts/serving_warmup.py").read_bytes()).hexdigest(), compat["current_sha256"])
        self.assertIn("55a7c4997782fb8fd2e12608249b5e0463604c17e0a15ec189d8c17e77c2ee85", compat["previous_sha256"])


if __name__ == "__main__":
    unittest.main()
