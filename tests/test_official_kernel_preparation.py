import importlib.metadata
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from official_kernel_preparation import chunk_bound, compile_family, prepare


class PreparationTests(unittest.TestCase):
    def test_chunk_bound_against_exhaustive_small_partitions(self):
        for size in (2, 3, 64):
            exact = {(0, 0): 0}
            for count in range(1, 9):
                for total in range(count, 81):
                    exact[count, total] = max(
                        exact[count - 1, total - length] + (length + size - 1) // size
                        for length in range(1, total + 1)
                        if (count - 1, total - length) in exact)
                    self.assertEqual(exact[count, total], chunk_bound(total, count, size))
        self.assertEqual(chunk_bound(4096, 24), 87)

    def test_compile_only_covers_observed_miss_and_full_bound(self):
        calls = []
        class Kernel:
            def __getitem__(self, grid):
                raise AssertionError('Device launch is forbidden in preparation')
            def warmup(self, *args, **kwargs):
                calls.append((args, kwargs))
                return SimpleNamespace(hash='cpu-fixture-' + str(kwargs['task_num']))
        dtypes = SimpleNamespace(bfloat16='bf16', float32='fp32', int32='i32')
        rows = compile_family(Kernel(), dtypes, tokens=4096, sequences=24, heads=16, key_heads=8, cores=24)
        self.assertEqual(len(rows), 87)
        self.assertEqual([row['task_num'] for row in rows], list(range(16, 1393, 16)))
        self.assertIn(80, [row['task_num'] for row in rows])
        for args, kwargs in calls:
            self.assertEqual(args, ('bf16', 'bf16', 'fp32', 'fp32', 'i32', 'i32'))
            self.assertEqual(kwargs['grid'], (24,))
            self.assertTrue(kwargs['IS_VARLEN'] and kwargs['USE_G'])
            self.assertEqual((kwargs['B'], kwargs['BT'], kwargs['K'], kwargs['BK']), (1, 64, 128, 128))

    def test_reject_invalid_scheduler_counts(self):
        for values in ((0, 24), (4096, -1), (True, 24), (4096.0, 24)):
            with self.assertRaises(ValueError):
                chunk_bound(*values)

    def test_first_compilation_cannot_start_after_guard(self):
        versions = {'vllm': '0.23.0+empty', 'vllm-ascend': '0.23.0', 'triton-ascend': '3.2.2'}
        with patch.object(importlib.metadata, 'version', side_effect=versions.__getitem__):
            with self.assertRaisesRegex(ValueError, 'must precede'):
                prepare(SimpleNamespace(_hust_compile_boundary=object()))

    def test_other_version_does_not_use_023_kernel_assumptions(self):
        with patch.object(importlib.metadata, 'version', return_value='0.30.0'):
            self.assertFalse(prepare(SimpleNamespace())['applicable'])


if __name__ == '__main__':
    unittest.main()
