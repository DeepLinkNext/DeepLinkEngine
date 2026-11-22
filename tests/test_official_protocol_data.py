"""CPU boundary checks. Full five-data/tokenizer preparation is also run on-host."""
from pathlib import Path
import hashlib,json,sys,tempfile,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from official import enable_module_imports

enable_module_imports()
import official_protocol_data as data

class Inputs(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
    def test_public_contract_retains_original_input_population(self):
        pin=data.contracts();self.assertEqual(pin['requests'],200);self.assertEqual(pin['input_tokens'],282265)
        self.assertEqual(pin['prompt_tokens_sha256'],'c36b966384a220b9060c452e632ac636da59a92dce5a6ded9fe931794c187851')
        self.assertEqual(pin['continuous_input_sha256'],'17faf7a6e4ba31f55fb64dca577d41d85ebc8ca4381ccf68d8184cc1e5937802')
    def test_missing_original_model_is_rejected_before_any_data_or_runtime(self):
        with self.assertRaisesRegex(ValueError,'missing'):data.admit_model(self.root,data.contracts())
        self.assertEqual(list(self.root.iterdir()),[])
    def test_actual_bytes_are_verified_and_wrong_source_is_preserved(self):
        source=self.root/'repo';p=source/'engine/models/qwen3.5-35b-a3b.json';p.parent.mkdir(parents=True)
        raw=b'{}';p.write_text(json.dumps(dict(files={'config.json':dict(bytes=2,sha256=hashlib.sha256(raw).hexdigest(),revision='a'*40)})))
        model=self.root/'model';model.mkdir();(model/'config.json').write_bytes(raw)
        pin=dict(model_manifest_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),tokenizer_files={'config.json':hashlib.sha256(raw).hexdigest()})
        with patch.object(data,'ROOT',source):
            self.assertEqual(data.admit_model(model,pin)['files_verified'],1)
            (model/'config.json').write_bytes(b'xx')
            with self.assertRaisesRegex(ValueError,'bytes differ'):data.admit_model(model,pin)
        self.assertEqual((model/'config.json').read_bytes(),b'xx')
    def test_wrong_official_anchor_is_rejected_before_full_source_read(self):
        anchor=self.root/'anchor';anchor.mkdir();(anchor/'image-lock.json').write_text('{}')
        with patch('official_image.file_map',side_effect=AssertionError('Wrong origin stops first')),self.assertRaisesRegex(ValueError,'actual selected official0.23'):data.admit_anchor(anchor)
    def test_wrong_protocol_profile_retains_failed_no_model_receipt(self):
        out=self.root/'out'
        with self.assertRaisesRegex(ValueError,'original full R8'):
            data.prepare(dict(out=str(out),model=str(self.root),deployment=str(self.root),engine_profile='r9',test_mode='full'),self.root)
        value=json.loads((out/'official-protocol-preparation.json').read_bytes());self.assertEqual(value['status'],'failed');self.assertFalse(value['model_service_started']);self.assertFalse(value['legacy018_sdk_admitted'])
    def test_public_flag_stops_before_full_run_policy(self):
        import run_all
        config=self.root/'config.json';model=self.root/'model';model.mkdir();config.write_text(json.dumps(dict(model=str(model),cache=str(self.root/'cache'),out=str(self.root/'new'),physical_devices=[0,1],engine_version='2026.11.17.8.post2',environment=dict(official_sdk=str(self.root/'anchor')))))
        args=['run_all.py','--config',str(config),'--prepare-official-protocol-only']
        with patch('official_sdk_bytecode.verify_or_repair',return_value=None),patch.object(sys,'argv',args),patch('official_execution.maybe_enter_config'),patch.object(run_all,'select_run_engine'),patch('engine.environment_setup.ensure_environment',return_value=None),patch.object(run_all,'preflight'),patch('benchmark.b0_run_policy.full_entry_policy',side_effect=AssertionError('No inference policy')),patch.object(run_all,'run',return_value=0) as execute:
            self.assertEqual(run_all.main(),0)
        self.assertIs(execute.call_args.kwargs['prepare_official_protocol_only'],True)

if __name__=='__main__':unittest.main()
