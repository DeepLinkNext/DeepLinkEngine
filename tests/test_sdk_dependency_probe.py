"""Real metadata subprocesses and declared client derivation; no native imports."""
import ast, hashlib, importlib.util, json, os
from pathlib import Path
import subprocess, sys, tempfile, unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from engine import sdk_dependency_probe as probe

class Admission(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
    def test_derivation_keeps_original_probe_and_rejects_other_changes(self):
        original=probe.original();derived,record=probe.derive(original)
        self.assertEqual(probe.probe(original),probe.probe(derived));self.assertTrue(record['unchanged_other_ast'])
        self.assertEqual(derived.replace(probe.NEW_COMMAND,probe.OLD_COMMAND),original)
        with self.assertRaises(ValueError):probe.derive(derived+b'\nother=True\n')
        with self.assertRaises(ValueError):probe.derive(original.replace(b'importlib.metadata,json,sys',b'importlib.metadata,json,os'))
    @unittest.skipUnless(sys.platform=='linux','Requires executable symlinks')
    def test_real_isolated_probe_and_failed_metadata_receipts(self):
        python=self.root/'venv/bin/python';python.parent.mkdir(parents=True);python.symlink_to(sys.executable)
        site=python.parent.parent/('lib/python%d.%d/site-packages'%sys.version_info[:2]);site.mkdir(parents=True)
        dist=site/'torch-9.1.dist-info';dist.mkdir();(dist/'METADATA').write_text('Name: torch\nVersion: 9.1\n')
        before={str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file() and not p.is_symlink()}
        receipt=self.root/'positive.json'
        with patch.dict(os.environ,PYTHONDONTWRITEBYTECODE='0'):
            actual=probe.inspect_sdk(python,{'torch':'9.1'},receipt=receipt)
        self.assertFalse(actual['native_packages_imported']);self.assertEqual(actual['metadata'],{'torch':'9.1'})
        self.assertEqual(json.loads(receipt.read_bytes())['command_flags'],['-I','-B','-S'])
        self.assertEqual({str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file() and not p.is_symlink() and p.suffix!='.json'},before)
        with self.assertRaisesRegex(ValueError,'metadata differs'):
            probe.inspect_sdk(python,{'torch':'9.2'},receipt=self.root/'negative.json')
        self.assertEqual(json.loads((self.root/'negative.json').read_bytes())['status'],'failed')
        raw=subprocess.check_output([str(python),'-I','-B','-S','-c','import json,sys;print(json.dumps([sys.flags.ignore_environment,sys.dont_write_bytecode]))'],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='0'),text=True)
        self.assertEqual(json.loads(raw),[1,True])
    def test_wrong_actual_inventory_rejected_before_spawn(self):
        tools=self.root/'tools';source=tools/probe.SOURCE;source.parent.mkdir(parents=True);source.write_bytes(probe.original())
        (tools/'DEPLOYMENT-TOOLS.json').write_text(json.dumps({'files':{probe.SOURCE:'0'*64}}))
        with patch.object(probe.subprocess,'run',side_effect=AssertionError('must not spawn')),self.assertRaisesRegex(ValueError,'actual tools inventory'):
            probe.inspect_sdk(sys.executable,{},tools=tools,receipt=self.root/'bad-source.json')
    @unittest.skipUnless(sys.platform=='linux','The public client uses Linux fcntl')
    def test_real_selected_r8_kit_recursive_import_and_selection_unchanged(self):
        from engine import release_selection as selection
        from benchmark import one_click
        from benchmark import evaluation_client
        record=dict(schema='deeplinkengine-stable-selection/v1',release='full-answer-b1-20261117-r8',legacy=True,engine_profile='r8',version='2026.11.17.8.post2',client=None)
        saved=self.root/'selection.json';saved.write_text(json.dumps(record));original=saved.read_bytes()
        with patch.dict(os.environ):
            os.environ.pop(selection.METADATA_CLIENT,None);selection.bind(saved)
            selected=selection.ensure_metadata_client(self.root/'cache');one_click.verify_kit()
            self.assertEqual(one_click.KIT,Path(selected['path']));self.assertEqual(saved.read_bytes(),original)
            files=evaluation_client.verify(selected)['files'];self.assertIn(probe.SOURCE,files)
            # These actual controller modules all resolve their normal
            # dependency_admission import from this explicit derived KIT.
            code='import importlib,json,sys;sys.path.insert(0,sys.argv[1]);names='+repr(['third_party','run_b1_r4','run_native_reference','run_canonical_reference','run_paired','engine_entry'])+';modules=[importlib.import_module(n) for n in names];import dependency_admission as d;print(json.dumps({"path":d.__file__,"probe":d.PROBE,"callers":[m.__name__ for m in modules]}))'
            result=subprocess.run([sys.executable,'-B','-c',code,str(one_click.KIT/'scripts')],capture_output=True,text=True,check=True)
            observed=json.loads(result.stdout);self.assertEqual(Path(observed['path']),one_click.KIT/probe.SOURCE);self.assertEqual(observed['probe'],probe.probe(probe.original()))
            self.assertEqual(len(observed['callers']),6)
            (one_click.KIT/probe.SOURCE).write_bytes(probe.original())
            with self.assertRaisesRegex(ValueError,'inventory changed'):selection.selected_client()
        one_click.KIT=ROOT/'benchmark/one-click-kit'

if __name__=='__main__':unittest.main()
