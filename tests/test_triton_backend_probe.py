"""Compiler identity fixtures; no device selection, model or kernel compilation."""
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT)]
import triton_backend_probe as probe
import environment_setup as setup


class Identity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.site = self.root / 'site'
        self.wheel = self.root / 'ascend-source.whl'
        rows = []
        self.modules = {}
        with zipfile.ZipFile(self.wheel, 'w') as archive:
            for module, name in probe.FILES.items():
                raw = ('original Ascend source fixture: ' + name).encode()
                path = self.site / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                archive.writestr(name, raw)
                digest = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b'=').decode()
                rows.append([name, 'sha256=' + digest, str(len(raw))])
                self.modules[module] = types.SimpleNamespace(__file__=str(path))
            prefix = 'triton_ascend-3.6.0.dist-info/'
            archive.writestr(prefix + 'METADATA', 'Metadata-Version: 2.1\nName: triton-ascend\nVersion: 3.6.0\n')
            archive.writestr(prefix + 'entry_points.txt', '[triton.backends]\nascend = triton.backends.ascend\n')
            output = io.StringIO()
            csv.writer(output).writerows(rows)
            archive.writestr(prefix + 'RECORD', output.getvalue())
        self.sha = hashlib.sha256(self.wheel.read_bytes()).hexdigest()
        self.modules['triton'].__version__ = '3.6.0'
        compiler = type('AscendBackend', (), {'__module__': 'triton.backends.ascend.compiler'})
        driver = type('AscendDriver', (), {'__module__': 'triton.backends.ascend.driver'})
        self.modules['triton.backends'].backends = dict(ascend=types.SimpleNamespace(compiler=compiler, driver=driver))
        self.modules['triton._C.libtriton'].ascend = types.SimpleNamespace(ir=object())
        self.modules['triton._C.libtriton'].buffer_ir = object()
        self.modules['triton.runtime.driver'].driver = types.SimpleNamespace(_default=None, _active=None)
        self.dist = types.SimpleNamespace(version='3.6.0', entry_points=[types.SimpleNamespace(group='triton.backends', name='ascend', value='triton.backends.ascend')], locate_file=lambda name: self.site / name, read_text=lambda name: output.getvalue() if name == 'RECORD' else None)

    def run_probe(self, *, load=None):
        with patch.object(probe.metadata, 'distribution', return_value=self.dist), patch.object(probe.metadata, 'version', return_value='3.6.0'):
            return probe.check(self.wheel, self.sha, load=load or self.modules.__getitem__)

    def test_original_wheel_bound_compiler_and_lazy_driver_pass(self):
        value = self.run_probe()
        self.assertTrue(value['verified'])
        self.assertFalse(value['driver_initialized'])
        self.assertFalse(value['kernel_compiled'])
        self.assertEqual(len(value['files']), len(probe.FILES))

    def test_same_metadata_with_plain_library_bytes_fails_before_import(self):
        (self.site / 'triton/_C/libtriton.so').write_bytes(b'plain Triton library')
        with self.assertRaisesRegex(ValueError, 'overwritten'), patch.object(probe.importlib, 'import_module', side_effect=AssertionError('No framework import')):
            self.run_probe(load=lambda name: self.fail('Wrong bytes must stop before import'))

    def test_missing_ascend_symbols_are_rejected(self):
        del self.modules['triton._C.libtriton'].buffer_ir
        with self.assertRaisesRegex(ValueError, 'Ascend compiler ABI'):
            self.run_probe()

    def test_wrong_entrypoint_or_compiler_origin_are_rejected(self):
        self.dist.entry_points[0].value = 'triton.backends.nvidia'
        with self.assertRaisesRegex(ValueError, 'entry point differs'):
            self.run_probe()
        self.dist.entry_points[0].value = 'triton.backends.ascend'
        self.modules['triton.backends.ascend.compiler'].__file__ = str(self.root / 'other.py')
        with self.assertRaisesRegex(ValueError, 'outside the locked files'):
            self.run_probe()

    def test_initialized_driver_is_not_cpu_only_qualification(self):
        self.modules['triton.runtime.driver'].driver._active = object()
        with self.assertRaisesRegex(ValueError, 'initialized a device driver'):
            self.run_probe()

    def test_wrong_source_archive_is_rejected_before_any_import(self):
        self.wheel.write_bytes(b'changed archive')
        with self.assertRaisesRegex(ValueError, 'selected lock'):
            self.run_probe(load=lambda name: self.fail('No import on wrong source'))

    def test_missing_cached_probe_runs_only_cpu_and_healthy_probe_is_reused(self):
        directory = self.root / 'environment'
        directory.mkdir()
        source = dict(filename=self.wheel.name, sha256=self.sha)
        integrity = dict(packages={'triton-ascend': dict(record_sha256=hashlib.sha256(self.dist.read_text('RECORD').encode()).hexdigest())})
        invocations = []
        def complete(argv, **kwargs):
            invocations.append(argv)
            self.assertNotIn('pip', argv)
            self.assertEqual(kwargs['env']['TORCH_DEVICE_BACKEND_AUTOLOAD'], '0')
            self.assertEqual(kwargs['env']['ASCEND_RT_VISIBLE_DEVICES'], '')
            Path(argv[argv.index('--out') + 1]).write_text(json.dumps(self.run_probe()))
            return types.SimpleNamespace(returncode=0)
        with patch.object(setup, 'runtime_lock', return_value=dict(environment_sources={'triton-ascend': source})), patch.object(setup, 'cached_wheel', return_value=self.wheel) as fetch, patch.object(setup.subprocess, 'run', side_effect=complete):
            first = setup.triton_backend_smoke(sys.executable, {'triton-ascend': '3.6.0'}, {}, dict(cache=str(self.root)), {}, directory, integrity)
            second = setup.triton_backend_smoke(sys.executable, {'triton-ascend': '3.6.0'}, {}, dict(cache=str(self.root)), {}, directory, integrity)
        self.assertFalse(first['reused'])
        self.assertTrue(second['reused'])
        self.assertEqual(len(invocations), 1)
        fetch.assert_called_once()

    def test_real_isolated_smoke_entry_loads_the_helper_before_frameworks(self):
        bad = self.root / 'wrong.whl'
        bad.write_bytes(b'wrong source')
        output = self.root / 'smoke.json'
        result = subprocess.run([sys.executable, '-I', '-B', str(ROOT / 'tools/runtime_import_smoke.py'), '--out', str(output), '--triton-wheel', str(bad), '--triton-sha256', '0' * 64], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 1)
        receipt = json.loads(output.read_bytes())
        self.assertFalse(receipt['model_loaded'])
        self.assertIn('selected lock', receipt['error'])
        self.assertNotIn('ModuleNotFoundError', receipt['error'])


if __name__ == '__main__':
    unittest.main()
