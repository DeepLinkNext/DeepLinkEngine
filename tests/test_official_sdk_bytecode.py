"""Real cached OCI bytecode recovery; no network or installed SDK fixture."""
import hashlib, io, json, os
from pathlib import Path
import sys, tarfile, tempfile, unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import official_sdk_bytecode as recovery
import official_image


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
        self.lock=dict(schema='deeplinkengine-official-baseline-image/v1',registry='https://quay.io/v2/ascend/vllm-ascend/',repository='vllm-project/vllm-ascend',platform=dict(os='linux',architecture='arm64'),upstream_version='0.23.0',ascend_version='0.23.0',manifest_digest='sha256:'+sha(b'{}'),config=dict(digest='sha256:'+sha(b'{}')),layers=[])
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
        for name in ('environment_setup.py','engine_install.py','pip_transport.py','official_build_assets.py','official_image.py'):
            tree=ast.parse((Path(__file__).resolve().parents[1]/name).read_text())
            for node in ast.walk(tree):
                if isinstance(node,(ast.List,ast.Tuple)):
                    strings=[n.value for n in node.elts if isinstance(n,ast.Constant) and isinstance(n.value,str)]
                    if '-I' in strings or '-S' in strings:
                        self.assertIn('-B',strings,(name,node.lineno))

    def test_actual_probe_disables_bytecode_even_when_environment_requests_writes(self):
        import environment_setup
        result=environment_setup.probe(sys.executable,[],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='0'))
        self.assertTrue(result['bytecode_writes_disabled']);self.assertEqual(result['base_prefix'],sys.base_prefix)

    def test_changed_installed_manifest_is_refused_before_replay(self):
        (self.sdk/'installed-files.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'unchanged original'):self.run_repair()

if __name__=='__main__':unittest.main()
