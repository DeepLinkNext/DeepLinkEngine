"""CPU public-entry and pre-exec identity tests; no real SDK qualification."""
import hashlib,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import native_sdk_controller as controller

class Native(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
    def sdk(self):
        sdk=self.root/'sdk';binary=sdk/'rootfs/usr/local/python3.12/bin/python3';binary.parent.mkdir(parents=True);binary.write_bytes(b'fixture binary, never executed')
        lib=binary.parent.parent/'lib/libpython3.12.so.1.0';lib.parent.mkdir();lib.write_bytes(b'fixture library')
        lock=(ROOT/'engine/official-baselines/0.23.0.json').read_bytes();(sdk/'image-lock.json').write_bytes(lock)
        files={p.relative_to(sdk).as_posix():dict(bytes=p.stat().st_size,sha256=controller.digest(p)) for p in (binary,lib)}
        (sdk/'installed-files.json').write_text(json.dumps(files))
        (sdk/'installation.json').write_text(json.dumps(dict(schema='deeplinkengine-official-sdk/v1',image_manifest=json.loads(lock)['manifest_digest'],installed_files_sha256=controller.digest(sdk/'installed-files.json'))))
        return sdk,binary
    def test_bootstrap_hashes_binary_library_and_rejects_bad_identity(self):
        sdk,python=self.sdk();result=controller.admit_interpreter(python)
        self.assertEqual(len(result['binary_and_libpython_sha256']),2)
        python.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'SHA differs'):controller.admit_interpreter(python)
        unrelated=self.root/'foreign/python';unrelated.parent.mkdir();unrelated.write_bytes(b'not SDK')
        with self.assertRaisesRegex(ValueError,'one admitted'):controller.admit_interpreter(unrelated)
    def test_handoff_uses_same_script_arguments_B_and_preserves_native_identity(self):
        sdk,python=self.sdk();env={}
        with patch.object(controller,'native_scope',return_value=True),patch.object(controller,'runtime_version',return_value=(3,10)),patch.object(controller,'state',return_value=dict(libc='2.38',root_dev_inode=[1,2],os_release_sha256='a'*64,pid=123,start_ticks='9')),patch('official_image.environment',return_value=(python,env)),patch('transport_trust.configure'),patch.object(controller.os,'execve',side_effect=RuntimeError('fixture exec boundary')) as execute:
            with self.assertRaisesRegex(RuntimeError,'exec boundary'):controller.maybe_enter(sdk,ROOT/'run_all.py',['--config','/x/config.json','--prepare-environment'],cache=self.root/'cache')
        command=execute.call_args.args[1];self.assertEqual(command,[str(python),'-B',str(ROOT/'run_all.py'),'--config','/x/config.json','--prepare-environment'])
        value=json.loads(env[controller.ENV]);self.assertFalse(value['chroot_performed']);self.assertFalse(value['mount_performed']);self.assertEqual(value['libc'],'2.38')
    def test_direct_installed_client_rejects_foreign_python_before_exec(self):
        sdk,python=self.sdk();deployment=self.root/'install/sdk';deployment.mkdir(parents=True)
        (deployment/'deployment.json').write_text(json.dumps(dict(python=str(python),root=str(deployment),release='r8')))
        (deployment.parent/'installation.json').write_text(json.dumps(dict(python='/foreign/python',deployment=str(deployment),release='r8',dependencies=dict(python=str(python),python_version='3.12.13'))))
        with patch.object(controller,'native_scope',return_value=True),patch.object(controller,'runtime_version',return_value=(3,10)),patch.object(controller.os,'execve',side_effect=AssertionError('must not exec')),self.assertRaisesRegex(ValueError,'original installation receipt'):
            controller.maybe_enter_deployment(deployment,ROOT/'one_click.py',['run'])
    def test_direct_compute_handoff_binds_original_installation_and_native_python(self):
        sdk,python=self.sdk();deployment=self.root/'installed/sdk';deployment.mkdir(parents=True)
        (deployment/'deployment.json').write_text(json.dumps(dict(python=str(python),root=str(deployment.resolve()),release='r8')))
        (deployment.parent/'installation.json').write_text(json.dumps(dict(python=str(python),deployment=str(deployment.resolve()),release='r8',dependencies=dict(python=str(python),python_version='3.12.13'))))
        with patch.object(controller,'native_scope',return_value=True),patch.object(controller,'runtime_version',return_value=(3,10)),patch.object(controller,'state',return_value=dict(libc='2.38',root_dev_inode=[1,2],os_release_sha256='a'*64,pid=123,start_ticks='9')),patch.object(controller.os,'execve',side_effect=RuntimeError('fixture exec boundary')) as execute:
            with self.assertRaisesRegex(RuntimeError,'exec boundary'):controller.maybe_enter_deployment(deployment,ROOT/'one_click.py',['run','--compute-reference-only'])
        _,command,env=execute.call_args.args
        self.assertEqual(command,[str(python),'-B',str(ROOT/'one_click.py'),'run','--compute-reference-only'])
        self.assertTrue(env['LD_LIBRARY_PATH'].startswith(str(python.resolve().parent.parent/'lib')))
        self.assertEqual(len(json.loads(env[controller.ENV])['bootstrap_admission']['binary_and_libpython_sha256']),2)
    @unittest.skipUnless(sys.platform=='linux','Linux real SDK-compatible metadata subprocess')
    def test_public_prepare_environment_reaches_real_isolated_probe_capless(self):
        import run_all, sdk_dependency_probe as probe
        python=self.root/'venv/bin/python';python.parent.mkdir(parents=True);python.symlink_to(sys.executable)
        site=python.parent.parent/('lib/python%d.%d/site-packages'%sys.version_info[:2]);site.mkdir(parents=True)
        model=self.root/'model';model.mkdir();config=self.root/'evaluation.json'
        config.write_text(json.dumps(dict(model=str(model),cache=str(self.root/'cache'),out=str(self.root/'out'),physical_devices=[0,1],environment=dict(official_sdk=str(self.root/'fake-sdk')))))
        calls=[]
        def ensure(data):
            observed=probe.inspect_sdk(python,{},receipt=self.root/'public-probe.json');calls.append(observed);return None
        with patch.object(sys,'argv',['run_all.py','--config',str(config),'--prepare-environment']),patch('official_sdk_bytecode.verify_or_repair',return_value=None),patch.dict(os.environ),patch.object(run_all,'select_run_engine'),patch('environment_setup.ensure_environment',side_effect=ensure),patch.object(run_all,'preflight'):
            os.environ.pop('DEEPLINK_EXECUTION_CONTEXT',None)
            self.assertEqual(run_all.main(),0)
        self.assertEqual(len(calls),1);self.assertFalse(calls[0]['native_packages_imported']);self.assertFalse((self.root/'out').exists())
        self.assertEqual(json.loads((self.root/'public-probe.json').read_bytes())['status'],'completed')

if __name__=='__main__':unittest.main()
