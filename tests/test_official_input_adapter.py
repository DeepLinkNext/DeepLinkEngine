"""The capless adapter binds one path, preserving the complete remaining AST."""
import ast,hashlib,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import official_input_adapter as adapter

RECIPE=b"from transformers import AutoTokenizer\nfrom pathlib import Path\nimport json\ntokenizer=AutoTokenizer.from_pretrained('/models/Qwen3.5-35B-A3B',local_files_only=True)\npath=Path(__file__).parent/'continuous-inputs-r1.json'\nwith path.open('x') as f:\n json.dump({'fixture':tokenizer.encode('x')},f)\nfor arm,out in []:\n raise AssertionError('historical execution must not run')\n"

class Binding(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
    def test_exact_one_path_and_remaining_ast_unchanged(self):
        derived=adapter.bind(RECIPE,self.root,adapter.sha(RECIPE));self.assertTrue(adapter.verify_binding(RECIPE,derived,self.root,adapter.sha(RECIPE)))
        with self.assertRaisesRegex(ValueError,'another AST'):adapter.verify_binding(RECIPE,derived+b'other=True\n',self.root,adapter.sha(RECIPE))
    def test_zero_or_two_original_calls_and_wrong_source_refused(self):
        for raw in [RECIPE.replace(b'AutoTokenizer.from_pretrained',b'Something.from_pretrained'),RECIPE.replace(b'path=Path',b"second=AutoTokenizer.from_pretrained('/models/Qwen3.5-35B-A3B')\npath=Path")]:
            with self.assertRaisesRegex(ValueError,'exactly one'):adapter.bind(raw,self.root,adapter.sha(raw))
        with self.assertRaisesRegex(ValueError,'source SHA'):adapter.bind(RECIPE,self.root,'0'*64)
    def test_model_sha_and_stat_binding_then_real_prefix_execution(self):
        repo=self.root/'repo';manifest=repo/'engine/models/qwen3.5-35b-a3b.json';manifest.parent.mkdir(parents=True)
        model=self.root/'model';model.mkdir();file=model/'config.json';file.write_bytes(b'{}')
        manifest.write_text(json.dumps(dict(files={'config.json':dict(bytes=2,sha256=adapter.sha(b'{}'),revision='a'*40)})))
        info=file.stat();admission=dict(manifest_sha256=adapter.sha(manifest.read_bytes()),files_verified=1,files={'config.json':dict(bytes=2,sha256=adapter.sha(b'{}'),revision='a'*40,stat_identity=[info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns])})
        pin=dict(model_manifest_sha256=admission['manifest_sha256'],continuous_recipe_sha256=adapter.sha(RECIPE),continuous_input_sha256=adapter.sha(json.dumps({'fixture':[7]}).encode()))
        sdk=self.root/'sdk';source=sdk/'runtime/entrypoints/prepare_continuous.py';source.parent.mkdir(parents=True);source.write_bytes(RECIPE)
        import types
        class Tokenizer:
            @staticmethod
            def from_pretrained(path,**kwargs):
                self.assertEqual(Path(path),model.resolve());return types.SimpleNamespace(encode=lambda text:[7])
        with patch.object(adapter,'ROOT',repo),patch.dict(sys.modules,transformers=types.SimpleNamespace(AutoTokenizer=Tokenizer)):
            result=adapter.prepare(sdk,self.root/'inputs',model,pin,admission);adapter.verify_record(result)
            self.assertEqual(source.read_bytes(),RECIPE);self.assertTrue(result['path_binding_adapter']['remaining_ast_identical'])
            admission['files']['config.json']['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'full SHA admission|differs'):adapter.validate_model(model,pin,admission)

if __name__=='__main__':unittest.main()
