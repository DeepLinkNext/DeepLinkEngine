import hashlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'benchmark/one-click-kit'
sys.path[:0] = [str(ROOT), str(TOOLS / 'scripts')]
from official import enable_module_imports

enable_module_imports()
from official_worker import inventory_collector


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = TOOLS / 'third-party-test-kit/proc_capture'
        self.path = self.directory / 'm01_model_inventory.py'
        self.record = dict(path=str(self.path), sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_loads_complete_helper_without_exposing_plugin_directory(self):
        plugins = {entry.name for distribution in importlib.metadata.distributions(path=[str(self.directory)])
                   for entry in distribution.entry_points if entry.group == 'vllm.general_plugins'}
        self.assertIn('hust_m01_proc_capture', plugins)
        before = list(sys.path)
        self.assertTrue(callable(inventory_collector(self.record)))
        self.assertEqual(sys.path, before)
        self.assertNotIn(str(self.directory), sys.path)

    def test_rejects_modified_helper_before_execution(self):
        with self.assertRaisesRegex(ValueError, 'changed'):
            inventory_collector(dict(self.record, sha256='0' * 64))

    def test_rejects_unbound_source_record(self):
        with self.assertRaisesRegex(ValueError, 'pinned'):
            inventory_collector(dict(path=str(self.path)))

    def test_discovered_legacy_plugin_is_a_noop_when_not_enabled(self):
        spec = importlib.util.spec_from_file_location('disabled_capture_cpu_review', self.directory / 'm01_proc_plugin.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        before = {name for name in sys.modules if name.startswith('vllm')}
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(module.install(), dict(plugin=module.PLUGIN, enabled=False, installed=False))
        self.assertEqual({name for name in sys.modules if name.startswith('vllm')}, before)


if __name__ == '__main__':
    unittest.main()
