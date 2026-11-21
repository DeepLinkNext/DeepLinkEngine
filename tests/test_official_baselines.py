"""CPU checks for separate official identities and safe OCI restoration."""
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from official_baselines import configuration
from official_image import destination, extract, read_lock, environment


class OfficialBaselines(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def config(self, **changes):
        model = self.root / 'model'
        model.mkdir(exist_ok=True)
        data = dict(model='model', cache='cache', out='new', physical_devices=[0, 7], versions=['0.23.0'])
        data.update(changes)
        path = self.root / 'evaluation.json'
        path.write_text(json.dumps(data))
        return path

    def test_explicit_version_and_paths(self):
        value = configuration(self.config())
        self.assertEqual(value['versions'], ['0.23.0'])
        self.assertEqual(value['out'], str(self.root / 'new'))

    def test_no_historical_relabelling_or_unknown_versions(self):
        for value in [['0.18.0'], ['0.23.0', '0.23.0'], ['latest'], ['0.23']]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                configuration(self.config(versions=value))

    def test_reject_existing_or_dangling_output(self):
        (self.root / 'new').symlink_to(self.root / 'missing')
        with self.assertRaises(ValueError):
            configuration(self.config())

    def test_reject_duplicate_cards_and_unknown_options(self):
        for changes in [dict(physical_devices=[0, 0]), dict(physical_devices=[True, 7]), dict(enforce_eager=True)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                configuration(self.config(**changes))

    def test_public_dispatch_never_enters_original_pair(self):
        import run_all
        with patch.object(sys, 'argv', ['run_all.py', '--official-baselines', '--config', 'evaluation.json']), patch('official_baselines.main', return_value=7) as official:
            self.assertEqual(run_all.main(), 7)
            official.assert_called_once_with(['--config', 'evaluation.json'])

    def layer(self, rows):
        path = self.root / 'layer.tar'
        with tarfile.open(path, 'w') as archive:
            for name, raw, target in rows:
                item = tarfile.TarInfo(name)
                item.mode = 0o755
                if target is not None:
                    item.type = tarfile.SYMTYPE
                    item.linkname = target
                    archive.addfile(item)
                else:
                    item.size = len(raw)
                    archive.addfile(item, io.BytesIO(raw))
        return path

    def test_whiteout_and_exact_file_modes(self):
        rootfs = self.root / 'rootfs'
        (rootfs / 'usr/local').mkdir(parents=True)
        (rootfs / 'usr/local/old').write_bytes(b'previous')
        path = self.layer([('usr/local/.wh.old', b'', None), ('usr/local/new', b'exact', None)])
        extract(path, rootfs)
        self.assertFalse((rootfs / 'usr/local/old').exists())
        self.assertEqual((rootfs / 'usr/local/new').read_bytes(), b'exact')
        self.assertEqual((rootfs / 'usr/local/new').stat().st_mode & 0o777, 0o755)

    def test_symlink_escape_rejected(self):
        rootfs = self.root / 'rootfs'
        rootfs.mkdir()
        with self.assertRaises(ValueError):
            destination(rootfs, '../outside')
        (rootfs / 'escape').symlink_to(self.root)
        with self.assertRaises(ValueError):
            destination(rootfs, 'escape/outside')
        path = self.layer([('usr/local/escape', b'', '../../../../outside')])
        with self.assertRaises(ValueError):
            extract(path, rootfs)

    def test_runtime_environment_is_private_and_removes_optimized_flags(self):
        image = self.root / 'image'
        home = image / 'rootfs/usr/local/python3.12.13'
        (home / 'bin').mkdir(parents=True)
        (home / 'bin/python3').write_bytes(b'fixture')
        (image / 'image-config.json').write_text(json.dumps(dict(config=dict(Env=[
            'PATH=/usr/local/python3.12.13/bin:/usr/bin',
            'LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/cann-9.1.0/lib64',
            'ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0']))))
        with patch.dict(os.environ, dict(BETTERSCALE_EXPERIMENT='1', HUST_TEST='1', GITHUB_TOKEN='fixture-secret', VLLM_PLUGINS='hust_joint_packed')):
            python, env = environment(image, self.root / 'cache', devices='0,1')
        self.assertEqual(python, home / 'bin/python3')
        self.assertNotIn('GITHUB_TOKEN', env)
        self.assertNotIn('BETTERSCALE_EXPERIMENT', env)
        self.assertNotIn('HUST_TEST', env)
        self.assertEqual(env['VLLM_PLUGINS'], 'ascend,ascend_model_loader,ascend_kv_connector')
        self.assertIn('/usr/local/Ascend/driver/lib64', env['LD_LIBRARY_PATH'])
        self.assertEqual(env['ASCEND_HOME_PATH'], str(image / 'rootfs/usr/local/Ascend/cann-9.1.0'))


if __name__ == '__main__':
    unittest.main()
