"""Verify fast SDK placement, durable aliases, and failure-safe migration."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import runtime_storage as storage


class RuntimeStorageTests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);self.root=Path(t.name)
        self.logical=self.root/'public/installation';self.logical.parent.mkdir()
        self.fast=self.root/'fast/installation';self.fast.parent.mkdir()
        self.frozen=dict(version='2026.11.17.8.post2',release='full-answer-b1-20261117-r8',runtime_files=9235,
                         archives={'sdk-r8.tar.gz':dict(sha256='a'*64,bytes=124)})
        self.cli=types.SimpleNamespace(verify=lambda:self.frozen)
        self.modules={'vllm_hust_engine':types.ModuleType('vllm_hust_engine'),'vllm_hust_engine.cli':self.cli}

    def build(self,command):
        target=Path(command[-1]);target.mkdir()
        (target/'sdk').mkdir();(target/'tools').mkdir()
        (target/'installation.json').write_text(json.dumps(dict(release=self.frozen['release'],package_version=self.frozen['version'],
                 python=sys.executable,deployment=str(target/'sdk'))))
        return types.SimpleNamespace(returncode=0)

    def invoke(self,reuse=False,run=None,audit=None):
        with patch.dict(sys.modules,self.modules),patch.object(storage,'storage_target',return_value=self.fast), \
             patch.object(storage.subprocess,'run',side_effect=run or self.build), \
             patch.object(storage,'audit',side_effect=audit or (lambda target,frozen:json.loads((target/'installation.json').read_text()))):
            return storage.ensure(self.logical,reuse=reuse)

    def test_successful_local_install_publishes_a_durable_alias_and_source_identity(self):
        self.invoke()
        self.assertTrue(self.logical.is_symlink())
        self.assertEqual(self.logical.resolve(),self.fast)
        record=json.loads(storage.marker(self.logical).read_text())
        self.assertTrue(record['verified']);self.assertEqual(record['runtime_files'],9235)
        self.assertEqual(record['archives'],self.frozen['archives'])

    def test_existing_nfs_install_is_preserved_until_new_local_install_passes(self):
        self.logical.mkdir();(self.logical/'original').write_text('preserve')
        self.invoke(reuse=True)
        record=json.loads(storage.marker(self.logical).read_text());old=Path(record['archived_previous'])
        self.assertEqual((old/'original').read_text(),'preserve')
        self.assertEqual(self.logical.resolve(),self.fast)

    def test_failed_local_install_never_moves_the_previous_install_or_publishes_alias(self):
        self.logical.mkdir();(self.logical/'original').write_text('preserve')
        with self.assertRaisesRegex(RuntimeError,'installation failed'):
            self.invoke(reuse=True,run=lambda command:types.SimpleNamespace(returncode=17))
        self.assertFalse(self.logical.is_symlink());self.assertEqual((self.logical/'original').read_text(),'preserve')
        self.assertFalse(storage.marker(self.logical).exists())

    def test_failed_audit_never_publishes_a_new_alias(self):
        with self.assertRaisesRegex(ValueError,'source differs'):
            self.invoke(audit=lambda *args:(_ for _ in ()).throw(ValueError('source differs')))
        self.assertFalse(self.logical.exists());self.assertFalse(storage.marker(self.logical).exists())

    def test_repeated_verification_reuses_the_same_valid_local_runtime(self):
        self.invoke()
        with patch.dict(sys.modules,self.modules),patch.object(storage,'storage_target',return_value=self.fast), \
             patch.object(storage.subprocess,'run') as run,patch.object(storage,'audit',return_value={}):
            storage.ensure(self.logical,reuse=True)
        run.assert_not_called()

    def test_local_target_symlink_is_rejected(self):
        outside=self.root/'outside';outside.mkdir();self.fast.symlink_to(outside,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'must not be a symlink'):self.invoke()
        self.assertEqual(list(outside.iterdir()),[])

    def test_storage_key_uses_the_runtime_identity_and_places_only_sdk_on_fast_disk(self):
        base=self.root/'scratch';base.mkdir()
        with patch.dict(storage.os.environ,DEEPLINKENGINE_RUNTIME_CACHE=str(base)), \
             patch.object(storage,'filesystem',side_effect=lambda p:'nfs' if Path(p)==self.logical.parent else 'ext4'), \
             patch.object(storage.shutil,'disk_usage',return_value=types.SimpleNamespace(free=3*2**30)):
            target=storage.storage_target(self.logical,self.frozen)
            changed=storage.storage_target(self.logical,dict(self.frozen,version='new-packaging'))
        self.assertTrue(target.is_relative_to(base));self.assertNotEqual(target,changed)
        self.assertTrue(self.logical.parent.exists())


    def test_missing_ephemeral_runtime_is_reconstructed_using_recorded_python(self):
        self.logical.symlink_to(self.fast,target_is_directory=True)
        venv=self.root/'venv';(venv/'bin').mkdir(parents=True);(venv/'pyvenv.cfg').write_text('home=/python')
        python=venv/'bin/python'
        storage.marker(self.logical).write_text(json.dumps(dict(logical=str(self.logical),verified=True,python=str(python))))
        with patch('environment_setup.cann_environment',return_value=({'CANN':'ready'},{})), \
             patch.object(storage.subprocess,'run',return_value=types.SimpleNamespace(returncode=0)) as run:
            storage.restore_missing(self.logical)
        command=run.call_args.args[0]
        self.assertEqual(command[0],str(python));self.assertEqual(command[-1],'--reuse')
        self.assertEqual(run.call_args.kwargs['env'],{'CANN':'ready'})

    def test_unmanaged_dangling_link_is_never_reconstructed(self):
        self.logical.symlink_to(self.fast,target_is_directory=True)
        with patch.object(storage.subprocess,'run') as run:
            storage.restore_missing(self.logical)
        run.assert_not_called()


    def test_nfs_verification_can_complete_after_the_old_preparation_deadline(self):
        self.build(['install',str(self.fast)])
        self.cli.verify_tools=lambda tools:None
        def slow(command,**kwargs):
            if kwargs.get('timeout',0)<=180:raise storage.subprocess.TimeoutExpired(command,180)
            self.assertTrue(kwargs['check'])
            self.assertIn('verify_installed_sources',command[3])
            self.assertIn('admit_dependencies',command[3])
            return types.SimpleNamespace(stdout=json.dumps(dict(verified_runtime_files=9235)))
        self.cli.audit_install=lambda sdk,tools:slow(['old-audit'],timeout=180)
        with patch.dict(sys.modules,self.modules),patch.object(storage,'filesystem',return_value='nfs4'), \
             patch.object(storage.subprocess,'run',side_effect=slow):
            self.assertEqual(storage.audit(self.fast,self.frozen)['release'],self.frozen['release'])

    def test_source_verification_failure_is_never_retried_or_accepted(self):
        self.build(['install',str(self.fast)])
        self.cli.verify_tools=lambda tools:None
        def reject(*args,**kwargs):raise storage.subprocess.CalledProcessError(1,['verify-sources'])
        self.cli.audit_install=reject
        with patch.dict(sys.modules,self.modules),patch.object(storage,'filesystem',return_value='nfs4'), \
             patch.object(storage.subprocess,'run',side_effect=reject) as run:
            with self.assertRaises(storage.subprocess.CalledProcessError):storage.audit(self.fast,self.frozen)
        self.assertFalse(storage.marker(self.logical).exists())



if __name__=='__main__':unittest.main()
