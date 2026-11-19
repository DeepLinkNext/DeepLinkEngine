"""Named RPCs run on the effective worker and reject active accounting hooks."""

from pathlib import Path
from contextlib import contextmanager
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compile_boundary import WorkerExtension


class Runner:
    @contextmanager
    def execute_model(self):
        yield None

    def _build_attention_metadata(self):
        pass


class Worker:
    def __init__(self):
        self.rank = 1
        self.device = SimpleNamespace(index=1)
        self.model_runner = Runner()


class RuntimeIdentityTests(unittest.TestCase):
    def setUp(self):
        self.worker = Worker()
        self.fences = []
        torch = SimpleNamespace(npu=SimpleNamespace(synchronize=self.fences.append))
        self.patch = patch.dict(sys.modules, torch=torch)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_named_identity_returns_actual_worker_and_fence(self):
        result = WorkerExtension.hust_runtime_identity(self.worker)
        self.assertEqual(result["rank"], 1)
        self.assertEqual(result["worker"], Worker.__module__ + ".Worker")
        self.assertEqual(result["rpc_transport"], "named_worker_method")
        self.assertEqual(len(self.fences), 1)
        self.assertEqual(set(result["methods"]), {"execute_model", "_build_attention_metadata"})
        method = result["methods"]["execute_model"]
        self.assertEqual(method["source"], str(Path(__file__).resolve()))
        self.assertNotEqual(method["entry_source"], method["source"])

    def test_named_sync_records_effective_rank_and_device(self):
        result = WorkerExtension.hust_runtime_synchronize(self.worker)
        self.assertEqual(result, dict(rank=1, device=1))
        self.assertEqual(self.fences, [self.worker.device])

    def test_active_host_accounting_is_rejected(self):
        self.worker.model_runner._m01_host_prefill = SimpleNamespace(active=True)
        with self.assertRaisesRegex(ValueError, "Host prefill accounting"):
            WorkerExtension.hust_runtime_identity(self.worker)

    def test_instance_level_runner_observer_is_rejected(self):
        self.worker.model_runner.execute_model = lambda: None
        with self.assertRaisesRegex(ValueError, "instance-level"):
            WorkerExtension.hust_runtime_identity(self.worker)


if __name__ == "__main__":
    unittest.main()
