"""Relocation safety and reconstruction of an idle private Python environment."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine import python_storage as storage


class PythonStorage(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.cache=self.root/'cache';self.managed=self.cache/'environments/run/venv'
        (self.managed/'bin').mkdir(parents=True);(self.managed/'lib').mkdir()
        (self.managed/'pyvenv.cfg').write_text('home = /python\n')
        (self.managed/'lib/package.py').write_text('fixed byte identity\n')
        (self.managed/'bin/python').symlink_to(sys.executable)
        self.base=self.root/'local';self.base.mkdir()
        self.env=patch.dict(storage.os.environ,DEEPLINKENGINE_RUNTIME_CACHE=str(self.base));self.env.start();self.addCleanup(self.env.stop)
        self.fs=patch.object(storage,'filesystem',side_effect=lambda p:'nfs' if Path(p)==self.managed.parent else 'ext4');self.fs.start();self.addCleanup(self.fs.stop)
        self.users=patch.object(storage,'active_users',return_value=[]);self.users.start();self.addCleanup(self.users.stop)
        self.disk=patch.object(storage.shutil,'disk_usage',return_value=type('Usage',(),{'free':20*2**30})());self.disk.start();self.addCleanup(self.disk.stop)

    def test_same_python_path_and_all_file_bytes_are_preserved_with_durable_original(self):
        storage.ensure(self.managed)
        self.assertTrue(self.managed.is_symlink());record=json.loads(storage.marker(self.managed).read_text())
        self.assertEqual(record['files_copied'],3);self.assertTrue(record['relocation_verified'])
        self.assertEqual((self.managed/'lib/package.py').read_bytes(),b'fixed byte identity\n')
        self.assertEqual((Path(record['archived_previous'])/'lib/package.py').read_bytes(),b'fixed byte identity\n')
        self.assertTrue(storage.admitted(self.managed/'bin/python',self.cache))

    def test_owned_alias_survives_persistent_mount_type_change(self):
        storage.ensure(self.managed)
        with patch.object(storage, 'filesystem', return_value='xfs'):
            self.assertEqual(storage.ensure(self.managed), self.managed)
        self.assertEqual((self.managed/'lib/package.py').read_bytes(), b'fixed byte identity\n')

    def test_local_remount_recovers_missing_ephemeral_files_and_checks_original(self):
        storage.ensure(self.managed)
        record=json.loads(storage.marker(self.managed).read_text())
        shutil.rmtree(Path(record['physical']))
        with patch.object(storage, 'filesystem', return_value='xfs'):
            storage.ensure(self.managed)
        self.assertEqual((self.managed/'lib/package.py').read_bytes(), b'fixed byte identity\n')
        shutil.rmtree(Path(record['physical']))
        (Path(record['archived_previous'])/'lib/package.py').write_text('changed original')
        with patch.object(storage, 'filesystem', return_value='xfs'):
            with self.assertRaisesRegex(ValueError, 'original private Python hash differs'):
                storage.ensure(self.managed)

    def test_local_unmanaged_alias_still_fails(self):
        shutil.rmtree(self.managed)
        self.managed.symlink_to(self.base)
        with patch.object(storage, 'filesystem', return_value='xfs'):
            with self.assertRaisesRegex(ValueError, 'Unmanaged private Python symlink'):
                storage.ensure(self.managed)

    def test_ephemeral_loss_reconstructs_exact_files_from_the_persistent_original(self):
        storage.ensure(self.managed);record=json.loads(storage.marker(self.managed).read_text())
        shutil.rmtree(Path(record['physical']))
        self.assertFalse(self.managed.exists());storage.ensure(self.managed)
        self.assertEqual((self.managed/'lib/package.py').read_bytes(),b'fixed byte identity\n')

    def test_interrupted_alias_publication_recovers_using_its_durable_record(self):
        storage.ensure(self.managed);record=json.loads(storage.marker(self.managed).read_text())
        self.managed.unlink()
        self.assertTrue(storage.admitted(self.managed/'bin/python',self.cache))
        storage.ensure(self.managed)
        self.assertTrue(self.managed.is_symlink())
        self.assertEqual(self.managed.resolve(),Path(record['physical']))

    def test_changed_persistent_original_is_rejected_during_reconstruction(self):
        storage.ensure(self.managed);record=json.loads(storage.marker(self.managed).read_text())
        shutil.rmtree(Path(record['physical']));(Path(record['archived_previous'])/'lib/package.py').write_text('changed')
        with self.assertRaisesRegex(ValueError,'hash differs'):storage.ensure(self.managed)
        self.assertFalse(self.managed.exists())

    def test_copy_failure_never_moves_the_previous_environment(self):
        with patch.object(storage,'copy_verified',side_effect=OSError('read failed')):
            with self.assertRaisesRegex(OSError,'read failed'):storage.ensure(self.managed)
        self.assertFalse(self.managed.is_symlink());self.assertTrue((self.managed/'pyvenv.cfg').is_file())

    def test_foreign_active_python_cannot_be_relocated(self):
        with patch.object(storage,'active_users',return_value=[999999]):
            with self.assertRaisesRegex(ValueError,'in use'):storage.ensure(self.managed,allow_current=True)
        self.assertFalse(self.managed.is_symlink())

    def test_unmanaged_symlink_is_rejected_without_mutation(self):
        shutil.rmtree(self.managed);outside=self.root/'outside';outside.mkdir();self.managed.symlink_to(outside)
        with self.assertRaisesRegex(ValueError,'Unmanaged'):storage.ensure(self.managed)
        self.assertEqual(list(outside.iterdir()),[])

    def test_existing_local_environment_needs_no_copy(self):
        with patch.object(storage,'filesystem',return_value='ext4'),patch.object(storage,'copy_verified') as copy:
            self.assertEqual(storage.ensure(self.managed),self.managed)
        copy.assert_not_called();self.assertFalse(self.managed.is_symlink())


    def test_automatic_no_disk_fallback_preserves_existing_nfs_environment(self):
        before=(self.managed/'lib/package.py').read_bytes()
        with patch.dict(storage.os.environ):
            storage.os.environ.pop('DEEPLINKENGINE_RUNTIME_CACHE',None)
            with patch.object(storage.shutil,'disk_usage',return_value=type('Usage',(),{'free':2**30})()):
                self.assertEqual(storage.ensure(self.managed),self.managed)
        self.assertFalse(self.managed.is_symlink())
        self.assertEqual((self.managed/'lib/package.py').read_bytes(),before)
        self.assertFalse(storage.marker(self.managed).exists())
        self.assertEqual(list(self.managed.parent.glob('venv-nfs-original-*')),[])

    def test_explicit_local_cache_without_capacity_is_rejected_without_mutation(self):
        with patch.object(storage.shutil,'disk_usage',return_value=type('Usage',(),{'free':2**30})()):
            with self.assertRaisesRegex(ValueError,'10GiB'):storage.ensure(self.managed)
        self.assertFalse(self.managed.is_symlink())
        self.assertFalse(storage.marker(self.managed).exists())

    def test_automatic_fallback_does_not_ignore_corrupt_storage_identity(self):
        storage.ensure(self.managed)
        marker=storage.marker(self.managed);record=json.loads(marker.read_text())
        record['logical']=str(self.root/'wrong-environment');marker.write_text(json.dumps(record))
        with patch.dict(storage.os.environ):
            storage.os.environ.pop('DEEPLINKENGINE_RUNTIME_CACHE',None)
            with patch.object(storage.shutil,'disk_usage',return_value=type('Usage',(),{'free':2**30})()):
                with self.assertRaisesRegex(ValueError,'identity differs'):storage.ensure(self.managed)
        self.assertTrue(self.managed.is_symlink())



if __name__=='__main__':unittest.main()
