import json
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from official import enable_module_imports

enable_module_imports()
from official_runtime_identity import runtime_identity, observer_basis


class RuntimeIdentityTests(unittest.TestCase):
    def test_released_image_keeps_its_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            sdk=Path(directory)
            (sdk/'image-lock.json').write_text(json.dumps(dict(manifest_digest='sha256:'+'a'*64)))
            value=runtime_identity(sdk)
            self.assertEqual(value['official_image_manifest'],'sha256:'+'a'*64)
            self.assertTrue(value['official_runtime'])
            self.assertNotIn('source_derivation_sha256',value)

    def fixture(self,sdk,status='prepared'):
        candidate=dict(base_image_manifest='sha256:'+'b'*64)
        lock=dict(manifest_kind='source-build-derivation',manifest_digest='sha256:'+'c'*64,
                  base_image_manifest=candidate['base_image_manifest'],git_tag_references={'vllm':'d'*40},
                  sdist_git_file_equivalence_certified=False)
        (sdk/'image-lock.json').write_text(json.dumps(lock))
        (sdk/'source-candidate-lock.json').write_text(json.dumps(candidate))
        (sdk/'source-build').mkdir()
        (sdk/'source-build/receipt.json').write_text(json.dumps(dict(status=status,candidate=candidate)))
        (sdk/'installation.json').write_text(json.dumps(dict(build_receipt_sha256=hashlib.sha256((sdk/'source-build/receipt.json').read_bytes()).hexdigest())))

    def test_derivation_and_base_are_never_reported_as_official_image(self):
        with tempfile.TemporaryDirectory() as directory:
            sdk=Path(directory);self.fixture(sdk)
            value=runtime_identity(sdk)
            self.assertEqual(value['runtime_kind'],'source-build-candidate')
            self.assertEqual(value['base_image_manifest'],'sha256:'+'b'*64)
            self.assertEqual(value['source_derivation_sha256'],'c'*64)
            self.assertEqual(len(value['build_receipt_sha256']),64)
            self.assertNotIn('official_image_manifest',value)
            self.assertFalse(value['official_runtime'])
            self.assertNotIn('Unmodified official image',observer_basis(value))

    def test_failed_candidate_cannot_produce_ready_runtime_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            sdk=Path(directory);self.fixture(sdk,'failed')
            with self.assertRaisesRegex(ValueError,'incomplete'):
                runtime_identity(sdk)

    def test_changed_receipt_cannot_be_relabelled_with_a_new_current_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            sdk=Path(directory);self.fixture(sdk)
            receipt=sdk/'source-build/receipt.json'
            value=json.loads(receipt.read_bytes());value['extra']='post-install change'
            receipt.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'incomplete or differs'):
                runtime_identity(sdk)

    def test_actual_serving_entry_rejects_wrong_runtime_before_framework_import(self):
        import official_serve
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);sdk=root/'sdk';sdk.mkdir()
            (sdk/'image-lock.json').write_text(json.dumps(dict(manifest_digest='sha256:'+'a'*64)))
            versions={name:'cpu-fixture' for name in ('vllm','vllm-ascend','torch','torch-npu')}
            plan=root/'plan.json';plan.write_text(json.dumps(dict(sdk=str(sdk),runtime_versions=versions,
                worker_sources={},runtime_identity={'runtime_kind':'wrong-cpu-fixture'})))
            with patch.object(official_serve.sys,'argv',['serve',str(plan)]), \
                 patch.object(official_serve.sys,'prefix',str(sdk)), \
                 patch.object(official_serve.metadata,'version',return_value='cpu-fixture'):
                with self.assertRaisesRegex(ValueError,'source/build identity differs'):
                    official_serve.main()


if __name__=='__main__':
    unittest.main()
