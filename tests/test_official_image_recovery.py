import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from official_image import quarantine_partial_sdk


class Recovery(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.lock=self.root/'requested.json';self.lock.write_bytes(b'official-lock')
        self.sdk=self.root/'sdk';self.sdk.mkdir();(self.sdk/'image-lock.json').write_bytes(self.lock.read_bytes())

    def test_lock_only_interruption_is_preserved_and_new_path_is_free(self):
        result=quarantine_partial_sdk(self.lock,self.sdk)
        self.assertFalse(self.sdk.exists())
        self.assertEqual((Path(result['previous_sdk'])/'image-lock.json').read_bytes(),b'official-lock')
        self.assertFalse(result['files_deleted'])
        self.assertTrue(Path(result['receipt']).is_file())

    def test_owned_partial_extraction_is_preserved_with_its_marker(self):
        marker=dict(schema='deeplinkengine-official-image-preparation/v1',destination=str(self.sdk.absolute()),lock_sha256=hashlib.sha256(self.lock.read_bytes()).hexdigest())
        (self.sdk/'preparation.json').write_text(json.dumps(marker));(self.sdk/'rootfs').mkdir();(self.sdk/'rootfs/partial').write_bytes(b'original')
        result=quarantine_partial_sdk(self.lock,self.sdk)
        self.assertEqual((Path(result['previous_sdk'])/'rootfs/partial').read_bytes(),b'original')

    def test_complete_extraction_is_left_for_metadata_only_recovery(self):
        (self.sdk/'installed-files.json').write_bytes(b'{}')
        self.assertIsNone(quarantine_partial_sdk(self.lock,self.sdk))
        self.assertTrue(self.sdk.exists())

    def test_wrong_lock_and_unknown_nonempty_directories_are_never_moved(self):
        for kind in ('wrong-lock','unknown-file','unowned-rootfs'):
            with self.subTest(kind=kind):
                sdk=self.root/kind;sdk.mkdir();(sdk/'image-lock.json').write_bytes(b'wrong' if kind=='wrong-lock' else self.lock.read_bytes())
                if kind=='unknown-file':(sdk/'notes.txt').write_bytes(b'keep')
                if kind=='unowned-rootfs':(sdk/'rootfs').mkdir()
                with self.assertRaises(ValueError):quarantine_partial_sdk(self.lock,sdk)
                self.assertTrue(sdk.exists())

    def test_wrong_marker_destination_is_never_moved(self):
        (self.sdk/'preparation.json').write_text(json.dumps(dict(schema='deeplinkengine-official-image-preparation/v1',destination='/different',lock_sha256=hashlib.sha256(self.lock.read_bytes()).hexdigest())))
        with self.assertRaises(ValueError):quarantine_partial_sdk(self.lock,self.sdk)
        self.assertTrue(self.sdk.exists())


if __name__=='__main__':unittest.main()
