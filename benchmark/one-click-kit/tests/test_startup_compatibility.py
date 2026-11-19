"""Exercise the preserved M01 wrapper together with compile-only startup."""

import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import startup_failure


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StartupCompatibilityTests(unittest.TestCase):
    def test_compile_only_private_runner_preserves_live_snapshot_guard(self):
        source = ROOT / "tests/fixtures/owned_optimistic_operand_r4.py"
        # Exact preserved r4 correction, not a mock of the snapshot policy.
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                         "1581d33ff9e5fa1a161102ec82330c48e12ef6b03480e12bdca8f69ca62e2ddf")
        correction = load("preserved_correction", source)

        class Tensor:
            dtype = "int64"

            def __init__(self, address):
                self.address = address

            def data_ptr(self):
                return self.address

        class Kernel:
            def __init__(self):
                self.compiles = 0

            def run(self, *args, **kwargs):
                if kwargs.get("warmup") is not True:
                    raise AssertionError("Synthetic startup must never launch a kernel")
                self.compiles += 1
                return SimpleNamespace(run=lambda: None)

        runner = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=16), device="npu",
            input_batch=SimpleNamespace(num_computed_tokens_cpu_tensor=Tensor(1)),
            num_computed_tokens=Tensor(2), _bs_fused_progress=False,
            _bs_fused_positions=True, use_async_spec_decode=True,
        )
        positions = Kernel()

        def prepare(current, n, total, incoming):
            self.assertIsNot(current, runner)
            self.assertIs(current.num_computed_tokens, runner.num_computed_tokens)
            positions.run(current.num_computed_tokens, incoming)
            return True

        position_reuse = SimpleNamespace(prepare=prepare, positions=positions)
        package = ModuleType("betterscale.models.qwen35")
        package.__path__ = []
        package.position_reuse = position_reuse
        package.token_progress = SimpleNamespace(correct=Kernel())
        package.state_commit = SimpleNamespace(commit=Kernel())
        package.state_slots = SimpleNamespace(publish=Kernel())
        fake_torch = SimpleNamespace(
            empty_like=lambda value: Tensor(3), empty=lambda *a, **k: Tensor(4),
            npu=SimpleNamespace(synchronize=lambda: None),
        )
        worker = SimpleNamespace(model_runner=runner)
        with patch.dict(sys.modules, {"torch": fake_torch, package.__name__: package}), patch.dict(os.environ, {}, clear=True):
            correction.install(worker)
            warmup = load(package.__name__ + ".serving_warmup", ROOT / "overlays/b1-r6/mods/betterscale/betterscale/models/qwen35/serving_warmup.py")
            before = dict(runner.__dict__)
            result = warmup.warmup(runner)
            self.assertEqual(result["positions"], 1)
            self.assertEqual(positions.compiles, 1)
            self.assertEqual(runner.__dict__, before)
            owned = worker._owned_optimistic_operand
            self.assertEqual(owned.calls, 0)
            self.assertFalse(owned.failed)
            self.assertNotIn("run", positions.__dict__)
            # The real request route is still fail-closed without a snapshot.
            with self.assertRaisesRegex(ValueError, "Active protected host"):
                position_reuse.prepare(runner, 1, 1, Tensor(5))
            self.assertTrue(owned.failed)

    def test_first_worker_error_beats_later_timeout_and_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "service.log"
            path.write_text(
                "WARNING kernel fallback\n"
                "(Worker_TP1 pid=42) ERROR 11-13 14:06:48 [multiproc_executor.py:1055] WorkerProc hit an exception.\n"
                "(Worker_TP1 pid=42) ERROR 11-13 14:06:48 [multiproc_executor.py:1055] Traceback (most recent call last):\n"
                "(Worker_TP1 pid=42) ERROR 11-13 14:06:48 [multiproc_executor.py:1055] ValueError: Active protected host progress snapshot required\n"
                "(EngineCore pid=50) ERROR 11-13 14:06:48 [core.py:1411] EngineCore failed to start.\n"
                "(APIServer pid=12) ERROR 11-13 14:39:00 [cleanup.py:1] RuntimeError: Terminated\n"
            )
            value = startup_failure.inspect(path)
            self.assertEqual(value["pid"], 42)
            self.assertEqual(value["root_error"], "ValueError: Active protected host progress snapshot required")
            self.assertEqual(value["line"], 2)

    def test_warnings_and_nonfatal_traceback_are_not_startup_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "service.log"
            path.write_text("WARNING kernel fallback\nTraceback from optional import\nValueError: expected\n")
            self.assertIsNone(startup_failure.inspect(path))


if __name__ == "__main__":
    unittest.main()
