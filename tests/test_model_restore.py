import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model_restore


class RestoreTests(unittest.TestCase):
    def test_existing_bytes_are_reverified_and_corruption_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            output = root / 'model'
            output.mkdir()
            (output / 'config.json').write_bytes(b'{}')
            lock = root / 'manifest.json'
            lock.write_text(json.dumps(dict(schema='deeplinkengine-model-download/v1', repository='Qwen/model',
                files={'config.json': dict(bytes=2, sha256=hashlib.sha256(b'{}').hexdigest(), revision='a' * 40)})))
            with patch('model_restore.subprocess.run', side_effect=AssertionError('Must reuse verified bytes')):
                self.assertEqual(model_restore.restore(lock, output)['status'], 'completed')
                (output / 'config.json').write_bytes(b'xx')
                with self.assertRaisesRegex(ValueError, 'Existing model file differs'):
                    model_restore.restore(lock, output)
                self.assertEqual((output / 'config.json').read_bytes(), b'xx')

    def test_unsafe_filename_and_mutable_revision_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'manifest.json'
            for name, revision in (('../escape', 'a' * 40), ('file', 'master')):
                p.write_text(json.dumps(dict(schema='deeplinkengine-model-download/v1', repository='Qwen/model',
                    files={name: dict(bytes=2, sha256='b' * 64, revision=revision)})))
                with self.assertRaises(ValueError):
                    model_restore.manifest(p)


if __name__ == '__main__':
    unittest.main()
