"""Exact producer CLI and real nested CPU probes; fixture SDKs are not engine qualification."""
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import producer_install_admission as adapter
import sdk_dependency_probe as probe

ORIGINAL = ROOT/'tests/fixtures/r8_producer_cli.py'


class Producer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw = ORIGINAL.read_bytes()

    def package(self):
        package = self.root/'package/vllm_hust_engine'
        package.mkdir(parents=True)
        (package/'cli.py').write_bytes(self.raw)
        (package/'__init__.py').write_text('__version__="2026.11.17.8.post2"\n')
        (package/'payload').mkdir()
        return package

    def test_exact_source_and_only_declared_ast_changes(self):
        payload = self.root/'original/payload'
        derived, record = adapter.derive(self.raw, payload=payload, receipts=self.root/'receipts')
        self.assertEqual(hashlib.sha256(self.raw).hexdigest(), adapter.ORIGINAL_SHA256)
        self.assertTrue(record['unchanged_other_ast'])
        self.assertIn('verify_installed_sources(root)', derived.decode())
        self.assertEqual(record['original_probe_sha256'], adapter.sha(probe.probe(probe.original()).encode()))
        tree = ast.parse(derived)
        audit = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'audit_install')
        code = next(n.value.value for n in audit.body if isinstance(n, ast.Assign) and n.targets[0].id == 'code')
        self.assertIn('tools=Path(sys.argv[2])', code)
        self.assertIn('receipt=receipt_path(', code)
        self.assertEqual(code.count('verify_installed_sources(root)'), 1)

    def test_unknown_source_is_rejected_before_materializing_or_execution(self):
        for raw in (self.raw+b'\nother=True\n', self.raw.replace(b'verify_installed_sources(root)', b'{}')):
            with self.assertRaisesRegex(ValueError, 'source requires review'):
                adapter.derive(raw, payload=self.root/'payload', receipts=self.root/'receipts')

    def command(self, package, python, target, evidence):
        cli = types.SimpleNamespace(__file__=str(package/'cli.py'))
        module = types.ModuleType('vllm_hust_engine')
        module.cli = cli
        with patch.dict(sys.modules, {'vllm_hust_engine':module}), patch.object(adapter.sys, 'executable', str(python)):
            return adapter.command(target, evidence)

    def test_materialized_source_has_honest_file_and_immutable_original(self):
        package = self.package()
        command = self.command(package, sys.executable, self.root/'new-sdk', self.root/'evidence')
        self.assertEqual(command[1:4], ['-I', '-B', '-c'])
        derived = Path(command[5])
        self.assertNotEqual(derived, package/'cli.py')
        self.assertEqual((package/'cli.py').read_bytes(), self.raw)
        value = json.loads(derived.with_suffix('.json').read_bytes())
        self.assertEqual(value['executed_source'], str(derived))
        self.assertEqual(value['payload_locator'], str(package/'payload'))
        self.assertFalse(value['original_package_modified'])
        self.assertFalse(value['original_tools_modified'])
        self.assertEqual(command, self.command(package, sys.executable, self.root/'new-sdk', self.root/'evidence'))
        derived.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'adapter differs'):
            self.command(package, sys.executable, self.root/'new-sdk', self.root/'evidence')

    def test_original_source_or_payload_symlink_is_rejected(self):
        if os.name == 'nt':
            self.skipTest('Requires unprivileged symlinks')
        package = self.package()
        source = package/'cli.py'
        source.unlink()
        source.symlink_to(ORIGINAL)
        with self.assertRaisesRegex(ValueError, 'ordinary installed'):
            self.command(package, sys.executable, self.root/'sdk', self.root/'evidence')

    def private_python(self, name):
        """Writable CPU-only stdlib copy: never run the negative against a real SDK."""
        home = self.root/name
        python = home/'bin/python3'
        python.parent.mkdir(parents=True)
        shutil.copy2(sys._base_executable, python)
        stdlib = home/('lib/python%d.%d'%sys.version_info[:2])
        shutil.copytree(sysconfig.get_path('stdlib'), stdlib,
            ignore=shutil.ignore_patterns('site-packages', 'dist-packages', '__pycache__',
                '*.pyc', 'test', 'tests', 'idlelib', 'tkinter', 'turtledemo', 'ensurepip'))
        site = stdlib/'site-packages/fixture-1.dist-info'
        site.mkdir(parents=True)
        (site/'METADATA').write_text('Name: fixture\nVersion: 1\n')
        return python, stdlib

    def payload(self, package, *, corrupt_source=False, wrong_metadata=False):
        tools = {'scripts/dependency_admission.py':probe.original(),
            'scripts/restore_b1_r4.py':b'''from pathlib import Path
def verify_installed_sources(root):
 if (Path(root)/'runtime/module.py').read_bytes()!=b'unchanged source':raise ValueError('source differs')
 return {'runtime/module.py':'fixture-only-source-verification'}
'''}
        tools['DEPLOYMENT-TOOLS.json'] = json.dumps({'files':{n:adapter.sha(v) for n,v in tools.items()}}).encode()
        sdk = {'deployment.json':json.dumps({'release':'full-answer-b1-20261117-r8'}).encode(),
            'runtime/module.py':b'corrupt source' if corrupt_source else b'unchanged source',
            'mods/microflow/hust_prefill_microflow/school028/host-contract.json':
                json.dumps({'versions':{'fixture':'2' if wrong_metadata else '1'}}).encode()}
        artifacts = {}
        for name, files, prefix in [('sdk-r8.tar.gz', sdk, ''), ('tools-r4.tar.gz', tools, 'vllm-hust-engine/')]:
            path = package/'payload'/name
            with tarfile.open(path, 'w:gz') as archive:
                for member, raw in files.items():
                    item = tarfile.TarInfo(prefix+member)
                    item.size = len(raw)
                    item.mode = 0o644
                    archive.addfile(item, io.BytesIO(raw))
            artifacts[name] = dict(bytes=path.stat().st_size, sha256=adapter.sha(path.read_bytes()))
        frozen = dict(release='full-answer-b1-20261117-r8', version='2026.11.17.8.post2',
            archives=artifacts, runtime_files=1, pythonpath_layout=['runtime'])
        (package/'payload/frozen.json').write_text(json.dumps(frozen))

    def run_install(self, python, package, target, evidence=None):
        # The tiny fixture claims no hardware ABI qualification. Only allow its
        # original producer's platform gate to reach the real CPU admission.
        prelude = 'import sys,platform;sys.path.insert(0,'+repr(str(package.parent))+');platform.machine=lambda:"aarch64";'
        if evidence is None:
            command = [str(python), '-I', '-B', '-c', prelude+
                'from vllm_hust_engine.cli import main;raise SystemExit(main())', 'install', '--out', str(target)]
        else:
            command = self.command(package, python, target, evidence)
            command[4] = prelude+command[4]
        return subprocess.run(command, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
            text=True, capture_output=True, timeout=90)

    @unittest.skipUnless(sys.platform == 'linux' and sys.version_info[:2] == (3,12), 'Real isolated producer fixture requires Linux CPython3.12')
    def test_real_producer_both_audits_no_pyc_and_original_control_does_write(self):
        package = self.package()
        self.payload(package)
        control, control_stdlib = self.private_python('control-python')
        original = self.run_install(control, package, self.root/'control-install')
        self.assertEqual(original.returncode, 0, original.stderr)
        self.assertGreater(len(list(control_stdlib.rglob('*.pyc'))), 0)
        python, stdlib = self.private_python('protected-python')
        before = {p.relative_to(package).as_posix():adapter.sha(p.read_bytes()) for p in package.rglob('*') if p.is_file()}
        result = self.run_install(python, package, self.root/'protected-install', self.root/'proof')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(list(stdlib.rglob('*.pyc')), [])
        self.assertEqual(before, {p.relative_to(package).as_posix():adapter.sha(p.read_bytes()) for p in package.rglob('*') if p.is_file()})
        receipts = [json.loads(p.read_bytes()) for p in (self.root/'proof/metadata/sdk-metadata-admissions').glob('*.json')]
        self.assertEqual(len(receipts), 2)
        self.assertTrue(all(r['status']=='completed' and r['command_flags']==['-I','-B','-S'] for r in receipts))
        self.assertEqual(json.loads((self.root/'protected-install/installation.json').read_bytes())['verified_runtime_files'], 1)

    @unittest.skipUnless(sys.platform == 'linux' and sys.version_info[:2] == (3,12), 'Real producer fixture requires Linux CPython3.12')
    def test_real_source_rejection_preserved_and_no_successful_installation(self):
        package = self.package()
        self.payload(package, corrupt_source=True)
        python, stdlib = self.private_python('python')
        result = self.run_install(python, package, self.root/'install', self.root/'proof')
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('verify_installed_sources(root)', result.stderr)
        self.assertFalse((self.root/'install').exists())
        self.assertEqual(list((self.root/'proof/metadata').rglob('*.json')), [])
        self.assertEqual(list(stdlib.rglob('*.pyc')), [])

    @unittest.skipUnless(sys.platform == 'linux' and sys.version_info[:2] == (3,12), 'Real producer fixture requires Linux CPython3.12')
    def test_real_metadata_rejection_preserved_with_failed_receipt(self):
        package = self.package()
        self.payload(package, wrong_metadata=True)
        python, stdlib = self.private_python('python')
        result = self.run_install(python, package, self.root/'install', self.root/'proof')
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse((self.root/'install').exists())
        receipts = list((self.root/'proof/metadata/sdk-metadata-admissions').glob('*.json'))
        self.assertEqual(len(receipts), 1)
        self.assertEqual(json.loads(receipts[0].read_bytes())['status'], 'failed')
        self.assertEqual(list(stdlib.rglob('*.pyc')), [])


if __name__ == '__main__':
    unittest.main()
