"""Exercise cross-mount admission without importing an accelerator runtime."""

import ast
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import serving_cache as cache


class ServingCacheTest(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "The serving runtime uses Linux file locks")
    def test_mount_change_imports_only_complete_graphs_and_kernel_groups(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old_pool = root / "old-mount"
            new_pool = root / "other-mount"
            identity = {"release": "r10", "runtime_config": "config", "hardware": "910B2"}
            src, status = cache.select(old_pool, None, identity, ["/old/sdk"])
            self.assertEqual(status["mode"], "cold")

            kernel_dir = src / cache.TRITON / "abc"
            kernel_dir.mkdir(parents=True)
            kernel = kernel_dir / "demo.bin"
            kernel.write_bytes(b"compiled kernel")
            group = kernel_dir / "__grp__demo.json"
            group.write_text(json.dumps({"child_paths": {kernel.name: str(kernel)}}))
            bad_group = kernel_dir / "__grp__bad.json"
            bad_group.write_text(json.dumps({"child_paths": {kernel.name: str(root / kernel.name)}}))

            draft = src / cache.VLLM / "aabb" / "rank_1_0" / "eagle_head"
            draft.mkdir(parents=True)
            graph = draft / "artifact_compile_range_1_6_subgraph_0"
            graph.write_text("def kernel(): return 42\n")
            process_local = draft / "artifact_compile_range_1_6_subgraph_1"
            process_local.write_text("triton_kernel_wrapper()\n")
            index = draft / "vllm_compile_cache.py"
            entries = {
                ((1, 6), i, "AscendCompiler"): {
                    "cache_key": None, "graph_handle": (artifact.name, str(artifact))
                }
                for i, artifact in enumerate((graph, process_local))
            }
            index.write_text(repr(entries))
            aot = src / cache.VLLM / "torch_aot_compile" / "opaque"
            aot.mkdir(parents=True)
            (aot / "model").write_text("opaque")

            shutil.move(old_pool, new_pool)
            dst, status = cache.select(new_pool, None, identity, ["/new/sdk"])
            self.assertEqual(status, {"mode": "relocated", "triton_groups": 1, "ascend_graphs": 1})
            copied = dst / graph.relative_to(src)
            self.assertEqual(copied.read_bytes(), (new_pool / graph.relative_to(old_pool)).read_bytes())
            rebased = ast.literal_eval((dst / index.relative_to(src)).read_text())
            self.assertEqual(list(rebased), [((1, 6), 0, "AscendCompiler")])
            self.assertEqual(rebased[((1, 6), 0, "AscendCompiler")]["graph_handle"][1], str(copied))
            group_path = dst / group.relative_to(src)
            self.assertEqual(json.loads(group_path.read_text())["child_paths"][kernel.name],
                             str(dst / kernel.relative_to(src)))
            self.assertFalse((dst / bad_group.relative_to(src)).exists())
            self.assertFalse((dst / process_local.relative_to(src)).exists())
            self.assertFalse((dst / aot.relative_to(src)).exists())
            self.assertEqual(ast.literal_eval((new_pool / index.relative_to(old_pool)).read_text()), entries)
            self.assertEqual(cache.select(new_pool, None, identity, ["/new/sdk"]),
                             (dst, {"mode": "existing"}))
            separate, status = cache.select(root / "another-pool", new_pool, identity, ["/new/sdk"])
            self.assertTrue(separate.is_dir())
            self.assertEqual(status, {"mode": "relocated", "triton_groups": 1, "ascend_graphs": 1})
            for key in ("release", "runtime_config", "hardware"):
                different = dict(identity, **{key: "different"})
                _, status = cache.select(new_pool, None, different, ["/new/sdk"])
                self.assertEqual(status["mode"], "cold")
            with cache.cache_lease(dst, writer=True) as writer:
                self.assertTrue(writer)
                with cache.cache_lease(dst, writer=False) as readable:
                    self.assertFalse(readable)
            (dst / cache.RECORD).write_text("invalid")
            isolated, status = cache.select(new_pool, None, identity, ["/new/sdk"])
            self.assertEqual(status["mode"], "isolated")
            self.assertNotEqual(isolated, dst)

    def test_device_fingerprint_uses_models_in_rank_order(self):
        sample = ("| npu-smi 26.0.rc1 Version: 26.0.rc1 |\n"
                  "| 4   910B2   | OK |\n"
                  "| 7   910C   | OK |\n")
        self.assertEqual(cache.parse_device_identity(sample, [7, 4]),
                         {"models": ["910C", "910B2"], "driver": "26.0.rc1"})
        with self.assertRaises(ValueError):
            cache.parse_device_identity(sample, [0, 4])


if __name__ == "__main__":
    unittest.main()
