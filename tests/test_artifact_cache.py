"""Known artifact cache hints must not write data or touch unrelated files."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine import artifact_cache


class ArtifactCacheTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.cache = self.root / 'cache'
        self.cache.mkdir()
        (self.root / 'engine').mkdir()
        (self.root / 'benchmark').mkdir()
        (self.root / 'engine/r8-lock.json').write_text('{}')
        (self.root / 'benchmark/one-click-lock.json').write_text('{}')
        self.size = 16 * 1024**2
        self.name = 'sha256-' + 'a' * 64
        self.path = self.cache / 'b0-oci' / self.name
        self.path.parent.mkdir()
        with self.path.open('wb') as f:
            f.write(b'known archive')
            f.truncate(self.size)
        self.data = dict(blobs=[dict(digest='sha256:' + 'a' * 64, bytes=self.size, parts=[])])

    def run_hint(self):
        with patch.object(artifact_cache, 'ROOT', self.root), \
             patch('benchmark.baseline_bundle.inventory', return_value=self.data), \
             patch('tools.verify_dependency_index.build_index', return_value={'artifacts': []}), \
             patch.object(artifact_cache, 'memory', return_value={'memory.current': 100, 'memory.max': 200}), \
             patch.dict(os.environ, {}, clear=True):
            return artifact_cache.release(self.cache)

    def test_missing_cgroup_counters_remain_explicit_in_diagnostics(self):
        with patch.object(Path, "read_text", side_effect=FileNotFoundError("no cgroup v2 counters")):
            result = artifact_cache.memory()
        self.assertIsNone(result["memory.current"])
        self.assertIn("no cgroup", result["read_error"])

    def test_unlimited_cgroup_is_recorded_without_inventing_a_finite_limit(self):
        with patch.object(Path, "read_text", side_effect=["123", "max"]):
            self.assertEqual(artifact_cache.memory(), {"memory.current": 123, "memory.max": None})

    def test_known_readonly_file_is_advised_without_changing_its_identity_or_bytes(self):
        before = self.path.stat()
        unrelated = self.path.parent / 'unrelated.bin'
        unrelated.write_bytes(b'do not touch')
        with patch.object(artifact_cache.os, 'posix_fadvise') as advise:
            result = self.run_hint()
        advise.assert_called_once()
        self.assertEqual(len(result['files']), 1)
        self.assertFalse(result['file_content_written'])
        self.assertFalse(result['global_drop_caches'])
        self.assertFalse(result['memory_limits_changed'])
        self.assertEqual(self.path.stat().st_mtime_ns, before.st_mtime_ns)
        self.assertEqual(self.path.read_bytes()[:13], b'known archive')
        self.assertEqual(unrelated.read_bytes(), b'do not touch')

    def test_size_mismatch_aborts_before_cache_hint(self):
        self.path.write_bytes(b'changed')
        with patch.object(artifact_cache.os, 'posix_fadvise') as advise:
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                self.run_hint()
        advise.assert_not_called()

    def test_selected_release_overrides_legacy_same_name_artifact_size(self):
        wheels=self.cache/'wheels';wheels.mkdir()
        name='vllm-same-version.whl';wheel=wheels/name
        with wheel.open('wb') as stream:stream.truncate(self.size+4)
        shared=wheels/'evaluation.tar.gz'
        with shared.open('wb') as stream:stream.truncate(self.size)
        records=[dict(name=name,bytes=self.size,kind='engine'),
                 dict(name=shared.name,bytes=self.size,kind='tools')]
        with patch.object(artifact_cache,'ROOT',self.root), \
             patch('benchmark.baseline_bundle.inventory',return_value=dict(blobs=[])), \
             patch('tools.verify_dependency_index.build_index',return_value=dict(artifacts=records)), \
             patch('engine.release_selection.active',return_value=dict(version='2026.11.19.9.post1')), \
             patch('engine.engine_install.sources',return_value=[dict(name=name,bytes=self.size+4)]), \
             patch.object(artifact_cache.os,'posix_fadvise') as advise:
            result=artifact_cache.release(self.cache)
        self.assertEqual(advise.call_count,2)
        self.assertEqual({Path(r['path']).name:r['bytes'] for r in result['files']},
                         {name:self.size+4,shared.name:self.size})
        self.assertFalse(result['file_content_written'])
        self.assertEqual(wheel.stat().st_size,self.size+4)

    def test_symlink_target_is_never_advised(self):
        target = self.root / 'outside'
        self.path.rename(target)
        self.path.symlink_to(target)
        with patch.object(artifact_cache.os, 'posix_fadvise') as advise:
            result = self.run_hint()
        advise.assert_not_called()
        self.assertEqual(len(result['hint_failures']), 1)

    def test_unsupported_filesystem_is_recorded_and_original_model_guard_remains_authoritative(self):
        with patch.object(artifact_cache.os, 'posix_fadvise', side_effect=OSError('unsupported')):
            result = self.run_hint()
        self.assertEqual(len(result['hint_failures']), 1)
        self.assertFalse(result['cache_reclamation_guaranteed'])


if __name__ == '__main__':
    unittest.main()
