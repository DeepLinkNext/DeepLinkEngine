import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] if Path(__file__).parent.name == 'tests' else Path(__file__).resolve().parent))
import official_source_candidate as candidate
import official_baselines as public_entry
import official_storage


class CandidateTests(unittest.TestCase):
    def setUp(self):
        storage = patch.object(official_storage, 'prefer', side_effect=lambda p: (Path(p), None))
        storage.start()
        self.addCleanup(storage.stop)
        # The local overlay need not copy the complete repository just for
        # controller fixtures. In CI this is the ordinary checkout root.
        source = patch.object(public_entry, 'ROOT', Path(official_storage.__file__).parent)
        source.start()
        self.addCleanup(source.stop)
        context = patch('official_execution.maybe_enter_official', return_value=None)
        context.start()
        self.addCleanup(context.stop)
        if sys.platform != 'linux':
            # The platform-mocked CLI fixture has no Linux /proc owner.
            owner = patch('official_sdk_state.owner_info', return_value={'cpu_fixture': True})
            owner.start()
            self.addCleanup(owner.stop)

    def archive(self, directory, names):
        path = Path(directory) / 'package.tar.gz'
        with tarfile.open(path, 'w:gz') as archive:
            for name, kind, body in names:
                member = tarfile.TarInfo(name)
                if kind == 'file':
                    member.size = len(body)
                    archive.addfile(member, io.BytesIO(body))
                else:
                    member.type = tarfile.SYMTYPE
                    member.linkname = body
                    archive.addfile(member)
        return path

    def test_complete_safe_sdist(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = self.archive(temp, [('p/setup.py', 'file', b'pass\n'), ('p/PKG-INFO', 'file', b'Version: 0.28.0\n')])
            out = Path(temp) / 'out'
            candidate.safe_extract(archive, out)
            self.assertEqual((out / 'setup.py').read_bytes(), b'pass\n')

    def test_escape_or_link_or_duplicate_rejected(self):
        for members in ([('../outside', 'file', b'x')], [('p/x', 'link', '../../outside')],
                        [('p/x', 'file', b'a'), ('p/x', 'file', b'b')],
                        [('p/x', 'file', b'a'), ('q/y', 'file', b'b')]):
            with self.subTest(members=members), tempfile.TemporaryDirectory() as temp:
                archive = self.archive(temp, members)
                with self.assertRaises(ValueError):
                    candidate.safe_extract(archive, Path(temp) / 'out')
                self.assertFalse((Path(temp) / 'outside').exists())

    def test_incomplete_sdist_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = self.archive(temp, [('p/setup.py', 'file', b'pass')])
            with self.assertRaises(ValueError):
                candidate.safe_extract(archive, Path(temp) / 'out')

    def test_explicit_source_identity_not_official_image(self):
        identity = candidate.identity()
        self.assertTrue(identity['candidate'])
        self.assertFalse(identity['official_release_image'])
        self.assertEqual(identity['ascend_version'], '0.28.0rc1')
        self.assertNotEqual(identity['version'], '0.28.0')
        self.assertFalse(identity['sdist_git_file_equivalence_certified'])
        self.assertEqual(len(candidate.key()), 64)

    def test_build_only_batch_override_is_absent_from_serving_environment(self):
        from official_image import environment
        import os
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);sdk=root/'sdk';home=sdk/'rootfs/usr/local/python3.12.13'
            (home/'bin').mkdir(parents=True);(home/'bin/python3').write_text('CPU fixture')
            (sdk/'image-config.json').write_text(json.dumps({'config':{'Env':[]}}))
            (sdk/'image-lock.json').write_text(json.dumps({'upstream_version':'0.28.0'}))
            with patch.dict(os.environ,{'VLLM_BATCH_INVARIANT':'1'}):
                _,env=environment(sdk,root/'cache')
            self.assertNotIn('VLLM_BATCH_INVARIANT',env)

    def test_remove_only_copied_editable_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            site = root / 'site'
            for package in ('vllm', 'vllm_ascend', 'torch'):
                d = site / (package + '-1.dist-info')
                d.mkdir(parents=True)
                (d / 'METADATA').write_text('Name: ' + package + '\n')
            (site / '__editable___vllm_023_finder.py').write_text('pass')
            removed = candidate.remove_old_editables(root, site)
            self.assertEqual(len(removed), 3)
            self.assertTrue((site / 'torch-1.dist-info').is_dir())
            self.assertFalse((site / 'vllm-1.dist-info').exists())

    def test_outside_package_destination_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'private').mkdir()
            (root / 'site').mkdir()
            with self.assertRaises(ValueError):
                candidate.remove_old_editables(root / 'private', root / 'site')

    def config(self, root, version):
        model = root / 'model'
        model.mkdir(exist_ok=True)
        path = root / 'evaluation.json'
        path.write_text(json.dumps(dict(model=str(model), cache=str(root / 'cache'),
            out=str(root / 'result'), physical_devices=[0, 1], versions=[version])))
        return path

    def test_public_candidate_prepare_uses_explicit_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = self.config(root, candidate.VERSION)
            with patch.object(public_entry.platform, 'system', return_value='Linux'), \
                 patch.object(public_entry.platform, 'machine', return_value='aarch64'), \
                 patch.object(candidate, 'prepare', return_value=dict(schema='fixture-only')) as build:
                code = public_entry.main(['--config', str(path), '--prepare-environment'])
            self.assertEqual(code, 0)
            build.assert_called_once()
            actual = json.loads((root / 'result/results.json').read_bytes())
            self.assertFalse(actual['model_execution_authorized'])
            self.assertEqual(actual['stages'][0]['baseline_id'], candidate.ID)
            self.assertFalse(actual['stages'][0]['official_release_image'])

    def test_public_candidate_failure_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = self.config(root, candidate.VERSION)
            with patch.object(public_entry.platform, 'system', return_value='Linux'), \
                 patch.object(public_entry.platform, 'machine', return_value='aarch64'), \
                 patch.object(candidate, 'prepare', side_effect=RuntimeError('controlled source build failure')):
                code = public_entry.main(['--config', str(path), '--prepare-environment'])
            self.assertEqual(code, 1)
            actual = json.loads((root / 'result/results.json').read_bytes())
            self.assertEqual(actual['status'], 'incomplete')
            self.assertEqual(actual['stages'][0]['status'], 'failed')
            self.assertIn('controlled source build failure', actual['stages'][0]['error'])

    def test_stable_028_still_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = self.config(root, '0.28.0')
            with patch.object(public_entry.platform, 'system', return_value='Linux'), \
                 patch.object(public_entry.platform, 'machine', return_value='aarch64'), \
                 patch.object(candidate, 'prepare') as build:
                self.assertEqual(public_entry.main(['--config', str(path), '--prepare-environment']), 1)
            build.assert_not_called()
            actual = json.loads((root / 'result/results.json').read_bytes())
            self.assertEqual(actual['stages'][0]['status'], 'blocked')

    @unittest.skipUnless(sys.platform == 'linux', 'Actual Linux owned-build process evidence')
    def test_real_failed_build_command_sealed_and_quiescent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(RuntimeError):
                candidate.command([sys.executable, '-B', '-c', 'import sys;print("controlled");sys.exit(7)'],
                                  dict(__import__('os').environ), root, root, 'fixture-failure', timeout=10)
            receipt = json.loads((root / 'fixture-failure.json').read_bytes())
            self.assertEqual(receipt['returncode'], 7)
            self.assertTrue(receipt['quiescent'])
            self.assertIn('controlled', (root / 'fixture-failure.log').read_text())


if __name__ == '__main__':
    unittest.main()
