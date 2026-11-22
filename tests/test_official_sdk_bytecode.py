"""Real cached OCI bytecode recovery; no network or installed SDK fixture."""
import hashlib, io, json, os
from pathlib import Path
import sys, tarfile, tempfile, unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from official import enable_module_imports, source_path

enable_module_imports()
import official_sdk_bytecode as recovery
import official_image
from official_sdk_state import refuse_active_owner as original_refuse_active_owner


def sha(raw): return hashlib.sha256(raw).hexdigest()
def row(raw): return dict(bytes=len(raw),sha256=sha(raw))

def inventory(sdk):
    return {p.relative_to(sdk).as_posix():row(p.read_bytes()) for p in (sdk/'rootfs').rglob('*') if p.is_file()}

class Recovery(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.sdk=self.root/'sdk';self.cache=self.root/'cache';self.cache.mkdir();self.sdk.mkdir()
        self.file='usr/local/python/lib/__pycache__/fixture.cpython-312.pyc';self.raw=b'original-bytecode'
        self.source='usr/local/python/lib/fixture.py'
        for name,raw in [(self.file,self.raw),(self.source,b'x=1\n')]:
            target=self.sdk/'rootfs'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        self.lock=dict(schema='deeplinkengine-official-baseline-image/v1',id='official-vllm-0.23.0',registry='https://quay.io/v2/ascend/vllm-ascend/',repository='vllm-project/vllm-ascend',platform=dict(os='linux',architecture='arm64'),upstream_version='0.23.0',ascend_version='0.23.0',manifest_digest='sha256:'+sha(b'{}'),config=dict(digest='sha256:'+sha(b'{}')),layers=[])
        self.layer({self.file:self.raw,self.source:b'x=1\n'})
        self.lockpath=self.root/'lock.json';self.lockpath.write_text(json.dumps(self.lock))
        for name,raw in [('image-lock.json',self.lockpath.read_bytes()),('image-manifest.json',b'{}'),('image-config.json',b'{}')]: (self.sdk/name).write_bytes(raw)
        (self.sdk/'installed-files.json').write_text(json.dumps(inventory(self.sdk)))
        (self.sdk/'installation.json').write_text(json.dumps(dict(schema='deeplinkengine-official-sdk/v1',image_manifest=self.lock['manifest_digest'],installed_files_sha256=sha((self.sdk/'installed-files.json').read_bytes()))))
        (self.sdk/'preparation.json').write_text('{}')
        self.original={p.name:p.read_bytes() for p in self.sdk.glob('*.json')}
        self.patches=[patch('official_image.file_map',side_effect=inventory),patch('official_sdk_state.refuse_active_owner',return_value={'stopped':True}),patch('official_sdk_state.owner_info',return_value={'pid':123}),patch.object(recovery,'users',return_value=[])]
        for item in self.patches:item.start();self.addCleanup(item.stop)
    def layer(self,files):
        stream=io.BytesIO()
        with tarfile.open(fileobj=stream,mode='w') as archive:
            for name,raw in files.items():
                info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
        raw=stream.getvalue();folder=self.cache/'official-oci';folder.mkdir(exist_ok=True);identity=sha(raw);(folder/('sha256-'+identity)).write_bytes(raw)
        self.lock['layers'].append(dict(digest='sha256:'+identity,size=len(raw)))
    def run_repair(self):return recovery.repair_locked(self.sdk,self.lockpath,self.cache)
    def completed_legacy(self, auxiliary=False):
        (self.sdk/'preparation.json').unlink()
        (self.sdk/'installation.json').unlink()
        prefix=self.sdk/'rootfs/usr/local/python'
        observed=dict(python=str(prefix/'bin/python3'),prefix=str(prefix),no_site=1,
            versions={'vllm':'0.23.0+empty','vllm-ascend':'0.23.0','torch':'fixture',
                      'torch-npu':'fixture','transformers':'fixture','triton-ascend':'fixture'},
            origins={name:str(prefix/'lib/python3.12/site-packages'/name/'__init__.py')
                     for name in ('torch','torch_npu','vllm','vllm_ascend')})
        raw=(self.sdk/'installed-files.json').read_bytes()
        official_image.installation_record(self.sdk,self.lock,raw,json.loads(raw),observed)
        if auxiliary:
            (self.sdk/'restoration-recovery.json').write_text(json.dumps(dict(
                all_original_extracted_files_reverified=True,source_code_patched=False,
                model_started=False,installed_files_sha256=sha(raw))))
        self.original={p.name:p.read_bytes() for p in self.sdk.glob('*.json')}

    def test_real_legacy_complete_restore_without_fabricating_owner_or_marker(self):
        self.completed_legacy(auxiliary=True)
        target=self.sdk/'rootfs'/self.file;target.write_bytes(b'legacy drift')
        result=self.run_repair();self.assertTrue(result['original_full_map_verified'])
        self.assertEqual(target.read_bytes(),self.raw);self.assertFalse((self.sdk/'preparation.json').exists())
        journal=Path(result['path']).parent;state=json.loads((journal/'result.json').read_bytes())['old_preparation_owner']
        self.assertEqual(state['mode'],'legacy-completed-sdk');self.assertFalse(state['past_preparation_owner_inferred'])
        self.assertEqual((journal/'original-drifted-bytecode/rootfs'/self.file).read_bytes(),b'legacy drift')
        for name,raw in self.original.items():self.assertEqual((self.sdk/name).read_bytes(),raw)

    def test_legacy_completion_does_not_require_auxiliary_recovery_receipt(self):
        self.completed_legacy();(self.sdk/'rootfs'/self.file).write_bytes(b'bad')
        self.assertTrue(self.run_repair()['original_full_map_verified'])
        self.assertFalse((self.sdk/'restoration-recovery.json').exists())

    def test_missing_marker_and_missing_installation_is_not_completed(self):
        self.completed_legacy();(self.sdk/'installation.json').unlink()
        target=self.sdk/'rootfs'/self.file;target.write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError,'ordinary original'):self.run_repair()
        self.assertEqual(target.read_bytes(),b'bad')

    def test_missing_marker_incomplete_baseline_or_outside_origins_are_refused(self):
        self.completed_legacy();target=self.sdk/'rootfs'/self.file;target.write_bytes(b'bad')
        original=json.loads((self.sdk/'installation.json').read_bytes())
        cases=[dict(baseline_id='other'),dict(files=0),dict(metadata={}),dict(model_started=True),
               dict(source_code_patched=True),dict(native_packages_imported=True)]
        import copy
        outside=copy.deepcopy(original['metadata']);outside['origins']['vllm']='/outside/vllm/__init__.py'
        cases.append(dict(metadata=outside))
        wrong=copy.deepcopy(original['metadata']);wrong['versions']['vllm']='0.30.0';cases.append(dict(metadata=wrong))
        for change in cases:
            with self.subTest(change=change):
                (self.sdk/'installation.json').write_text(json.dumps(dict(original,**change)))
                with self.assertRaises(ValueError):self.run_repair()
                self.assertEqual(target.read_bytes(),b'bad')
                self.assertFalse((self.cache/'sdk-bytecode-recovery').exists())

    def test_legacy_original_lock_config_and_manifest_must_still_match(self):
        self.completed_legacy();target=self.sdk/'rootfs'/self.file;target.write_bytes(b'bad')
        for name in ('image-lock.json','image-config.json','image-manifest.json','installed-files.json'):
            with self.subTest(name=name):
                before=(self.sdk/name).read_bytes();(self.sdk/name).write_bytes(b'{"not-original":true}')
                try:
                    with self.assertRaisesRegex(ValueError,'unchanged original'):self.run_repair()
                    self.assertEqual(target.read_bytes(),b'bad')
                finally:(self.sdk/name).write_bytes(before)

    def test_existing_unknown_and_live_owners_never_fall_back_to_legacy(self):
        self.completed_legacy();target=self.sdk/'rootfs'/self.file;target.write_bytes(b'bad')
        with patch('official_sdk_state.refuse_active_owner',side_effect=original_refuse_active_owner):
            (self.sdk/'preparation.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Unknown or legacy'):self.run_repair()
            owner=dict(schema='deeplinkengine-sdk-preparation-owner/v1',hostname='fixture',
                boot_id='00000000-0000-0000-0000-000000000001',pid=23,start_ticks=100)
            (self.sdk/'preparation.json').write_text(json.dumps(dict(owner=owner)))
            with patch('official_sdk_state._host_identity',return_value=dict(hostname='fixture',boot_id=owner['boot_id'])), \
                 patch('official_sdk_state._process_identity',return_value=dict(state='R',start_ticks=100)), \
                 self.assertRaisesRegex(ValueError,'still active'):self.run_repair()
        self.assertEqual(target.read_bytes(),b'bad');self.assertFalse((self.cache/'sdk-bytecode-recovery').exists())

    def test_legacy_completion_never_skips_real_zero_users_or_whole_oci_digest(self):
        self.completed_legacy();target=self.sdk/'rootfs'/self.file;target.write_bytes(b'bad')
        with patch.object(recovery,'users',return_value=[dict(pid=23)]),self.assertRaisesRegex(ValueError,'in use'):
            self.run_repair()
        self.assertFalse((self.cache/'sdk-bytecode-recovery').exists())
        next((self.cache/'official-oci').iterdir()).write_bytes(b'not the original SHA')
        with self.assertRaisesRegex(ValueError,'no download attempted'):self.run_repair()
        self.assertEqual(target.read_bytes(),b'bad')

    def test_legacy_completion_does_not_override_contradictory_auxiliary_metadata(self):
        self.completed_legacy(auxiliary=True);(self.sdk/'rootfs'/self.file).write_bytes(b'bad')
        (self.sdk/'restoration-recovery.json').write_text(json.dumps(dict(
            all_original_extracted_files_reverified=True,source_code_patched=False,
            model_started=False,installed_files_sha256='0'*64)))
        with self.assertRaisesRegex(ValueError,'auxiliary evidence differs'):self.run_repair()

    @unittest.skipUnless(sys.platform=='linux','Public SDK mutex uses Linux fcntl')
    def test_public_prepare_environment_recovers_legacy_before_any_private_sdk_exec(self):
        import run_all
        self.completed_legacy();target=self.sdk/'rootfs'/self.file;target.write_bytes(b'bad')
        locks=self.root/'engine/official-baselines';locks.mkdir(parents=True)
        (locks/'0.23.0.json').write_bytes(self.lockpath.read_bytes())
        model=self.root/'model';model.mkdir();config=self.root/'evaluation.json'
        config.write_text(json.dumps(dict(model=str(model),out=str(self.root/'out'),cache=str(self.cache),
            physical_devices=[0,1],environment=dict(official_sdk=str(self.sdk)))))
        observed=[]
        def recovered(*args,**kwargs):
            self.assertEqual(target.read_bytes(),self.raw);observed.append('public startup recovered');return False
        with patch.dict(os.environ),patch.object(sys,'argv',['run_all.py','--config',str(config),'--prepare-environment']), \
             patch.object(recovery,'ROOT',self.root),patch.object(run_all,'select_run_engine'),patch.object(run_all,'preflight'), \
             patch('engine.native_sdk_controller.maybe_enter',side_effect=recovered), \
             patch('official_execution.needed',return_value=False),patch('official_execution.maybe_enter_config'), \
             patch('engine.environment_setup.ensure_environment',return_value=None):
            self.assertEqual(run_all.main(),0)
        self.assertEqual(observed,['public startup recovered']);self.assertFalse((self.root/'out').exists())
        self.assertFalse((self.sdk/'preparation.json').exists())
    def test_real_restore_preserves_bad_bytes_and_original_manifest(self):
        target=self.sdk/'rootfs'/self.file;target.write_bytes(b'changed')
        result=self.run_repair();self.assertTrue(result['original_full_map_verified']);self.assertEqual(target.read_bytes(),self.raw)
        journal=Path(result['path']).parent
        self.assertEqual((journal/'original-drifted-bytecode/rootfs'/self.file).read_bytes(),b'changed')
        for name,raw in self.original.items():self.assertEqual((self.sdk/name).read_bytes(),raw)
    def test_healthy_cache_never_scans_users_or_replays_layers(self):
        with patch.object(recovery,'original_bytes',side_effect=AssertionError('no replay')),patch.object(recovery,'users',side_effect=AssertionError('no scan')):self.assertIsNone(self.run_repair())
    def test_non_bytecode_source_drift_is_preserved_and_refused(self):
        target=self.sdk/'rootfs'/self.source;target.write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError,'source/native'):self.run_repair()
        self.assertEqual(target.read_bytes(),b'bad');self.assertFalse((self.cache/'sdk-bytecode-recovery').exists())
    def test_extra_bytecode_requires_original_unchanged_source(self):
        expected={'rootfs/x.py':row(b'x')};observed=dict(expected,**{'rootfs/__pycache__/x.cpython-312.pyc':row(b'y')})
        self.assertEqual(len(recovery.classify(expected,observed)),1)
        with self.assertRaisesRegex(ValueError,'unchanged original source'):recovery.classify({},observed)
    def test_active_user_blocks_before_any_repair(self):
        target=self.sdk/'rootfs'/self.file;target.write_bytes(b'bad')
        with patch.object(recovery,'users',return_value=[{'pid':23}]),self.assertRaisesRegex(ValueError,'in use'):self.run_repair()
        self.assertEqual(target.read_bytes(),b'bad')
    def test_changed_original_blob_is_not_downloaded_or_committed(self):
        target=self.sdk/'rootfs'/self.file;target.write_bytes(b'bad')
        blob=next((self.cache/'official-oci').iterdir());blob.write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError,'no download attempted'):self.run_repair()
        self.assertEqual(target.read_bytes(),b'bad')
    def test_overlay_whiteout_cannot_resurrect_removed_bytecode(self):
        self.layer({str(Path(self.file).parent.as_posix())+'/.wh.'+Path(self.file).name:b''})
        with self.assertRaisesRegex(ValueError,'OCI replay'):recovery.original_bytes(self.lock,self.cache/'official-oci',{self.file:row(self.raw)})
    def test_later_layer_replaces_earlier_expected_bytecode(self):
        self.layer({self.file:b'new'})
        result,_=recovery.original_bytes(self.lock,self.cache/'official-oci',{self.file:row(b'new')})
        self.assertEqual(result[self.file],b'new')
    def test_production_isolated_python_commands_explicitly_disable_bytecode(self):
        import ast
        for name in ('engine/environment_setup.py', 'engine/engine_install.py', 'engine/pip_transport.py',
                     'official_build_assets.py', 'official_image.py'):
            tree=ast.parse(source_path(Path(__file__).resolve().parents[1], name).read_text())
            for node in ast.walk(tree):
                if isinstance(node,(ast.List,ast.Tuple)):
                    strings=[n.value for n in node.elts if isinstance(n,ast.Constant) and isinstance(n.value,str)]
                    if '-I' in strings or '-S' in strings:
                        self.assertIn('-B',strings,(name,node.lineno))

    def test_actual_probe_disables_bytecode_even_when_environment_requests_writes(self):
        from engine import environment_setup
        result=environment_setup.probe(sys.executable,[],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='0'))
        self.assertTrue(result['bytecode_writes_disabled']);self.assertEqual(result['base_prefix'],sys.base_prefix)

    def test_changed_installed_manifest_is_refused_before_replay(self):
        (self.sdk/'installed-files.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'unchanged original'):self.run_repair()

if __name__=='__main__':unittest.main()
