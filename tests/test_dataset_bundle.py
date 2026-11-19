"""Offline source restoration, integrity failures and archive isolation; no model inference."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import dataset_bundle as bundle


class DataBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.offline = self.root / 'archives'
        self.offline.mkdir()
        self.cache = self.root / 'cache'
        self.body = b'original question, answer, image and messages\n'
        self.file = 'hle/gold/test.jsonl'
        self.component = self.make_archive([(self.file, self.body)])
        self.manifest = dict(components={n: self.component for n in ('mmlu','hle','frontier','swe','terminal')})

    def make_archive(self, entries):
        path = self.offline / 'source.tar.gz'
        with tarfile.open(path, 'w:gz') as target:
            for name, raw in entries:
                item = tarfile.TarInfo(name); item.size = len(raw)
                target.addfile(item, io.BytesIO(raw))
        return dict(name=path.name, bytes=path.stat().st_size, sha256=bundle.digest(path),
            files={self.file: dict(bytes=len(self.body), sha256=hashlib.sha256(self.body).hexdigest())})

    def component_call(self):
        with patch.object(bundle, 'manifest', return_value=self.manifest), \
             patch.object(bundle.urllib.request, 'urlopen', side_effect=AssertionError('network')):
            return bundle.component(self.cache, 'hle', self.offline)

    def test_offline_original_restored_and_rechecked_without_network(self):
        source = self.component_call()
        self.assertEqual((source/self.file).read_bytes(), self.body)
        self.assertEqual(self.component_call(), source)

    def test_corrupt_archive_fails_before_any_data_is_installed(self):
        (self.offline / self.component['name']).write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'archive missing or changed'):
            self.component_call()
        self.assertFalse(list((self.cache/'deeplinkengine-data').glob('hle-*')))

    def test_changed_extracted_source_is_not_silently_reused(self):
        source = self.component_call()
        (source/self.file).write_bytes(b'altered')
        with self.assertRaisesRegex(ValueError, 'missing or changed'):
            self.component_call()

    def test_unlisted_traversal_is_rejected_even_with_valid_outer_hash(self):
        self.component = self.make_archive([('../escape', self.body)])
        self.manifest['components']['hle'] = self.component
        with self.assertRaisesRegex(ValueError, 'unsafe dataset archive member'):
            self.component_call()
        self.assertFalse((self.cache/'escape').exists())

    def test_duplicate_member_is_rejected(self):
        self.component = self.make_archive([(self.file,self.body),(self.file,self.body)])
        self.manifest['components']['hle'] = self.component
        with self.assertRaisesRegex(ValueError, 'unsafe dataset archive member'):
            self.component_call()

    def test_mismatched_asset_view_is_retained(self):
        source = self.component_call()
        dest = self.root/'assets';(dest/self.file).parent.mkdir(parents=True)
        (dest/self.file).write_bytes(b'user changed')
        with self.assertRaisesRegex(ValueError, 'original retained'):
            bundle.install_files(source, dest, self.component)
        self.assertEqual((dest/self.file).read_bytes(), b'user changed')

    def test_asset_view_cannot_mutate_sealed_component(self):
        source = self.component_call()
        dest = self.root/'assets';bundle.install_files(source,dest,self.component)
        (dest/self.file).write_bytes(b'changed view')
        self.assertEqual((source/self.file).read_bytes(), self.body)

    def test_existing_sources_and_sdk_work_without_archives_or_network(self):
        assets = self.root / 'supplied'
        (assets / self.file).parent.mkdir(parents=True)
        (assets / self.file).write_bytes(self.body)
        sdk = self.root / 'sdk'
        (sdk / 'inputs').mkdir(parents=True)
        (sdk / 'inputs/example.json').write_bytes(self.body)
        mmlu = dict(self.component, files={'mmlu/inputs/example.json':
            dict(bytes=len(self.body), sha256=hashlib.sha256(self.body).hexdigest())})
        self.manifest['components']['mmlu'] = mmlu
        with patch.object(bundle,'manifest',return_value=self.manifest), \
             patch.object(bundle,'component',side_effect=AssertionError('archive fetch')), \
             patch.object(bundle.urllib.request,'urlopen',side_effect=AssertionError('network')):
            view = bundle.prepare(self.cache,['hle'],assets=assets,deployment=sdk)
        self.assertEqual((view/self.file).read_bytes(),self.body)
        self.assertEqual((view/'mmlu/inputs/example.json').read_bytes(),self.body)
        self.assertFalse((assets/'mmlu').exists())
        self.assertEqual((assets/self.file).read_bytes(),self.body)

    def test_public_full_and_quick_dispatch_use_local_data(self):
        import one_click
        for mode, expected in [('mmlu',['mmlu']),('hle,frontier,swe,terminal',['hle','frontier','swe','terminal'])]:
            with patch.object(bundle,'prepare',return_value=self.root/'assets') as prepare, \
                 patch('baseline_bundle.ensure_deployment', return_value=Path('/baseline')), patch.object(one_click, 'validate_deployment'):
                args=one_click.prepare_data(['--cache',str(self.cache),'--deployment','/sdk','--datasets',mode])
            self.assertEqual(prepare.call_args.args[1],expected)
            self.assertEqual(args[-2:],['--dataset-assets',str(self.root/'assets')])
        with patch.object(bundle,'prepare',return_value=self.root/'assets') as prepare, \
             patch('baseline_bundle.ensure_deployment', return_value=Path('/baseline')), patch.object(one_click, 'validate_deployment'):
            args=one_click.prepare_data(['--context-tenant-only','--cache',str(self.cache),'--deployment','/sdk'])
        self.assertTrue(prepare.call_args.kwargs['context'])
        self.assertNotIn('--dataset-assets',args)

    def test_old_engine_release_is_rejected_before_any_large_input_download(self):
        import one_click
        sdk = self.root / 'old-sdk'
        sdk.mkdir()
        (sdk / 'deployment.json').write_text(json.dumps({'release':'full-answer-b1-20261030-r4'}))
        with patch('baseline_bundle.ensure_deployment') as baseline, patch.object(bundle, 'prepare') as data:
            with self.assertRaisesRegex(ValueError, 'installed frozen r8 SDK'):
                one_click.prepare_data(['--cache',str(self.cache),'--deployment',str(sdk)])
        baseline.assert_not_called()
        data.assert_not_called()

if __name__ == '__main__':
    unittest.main()
