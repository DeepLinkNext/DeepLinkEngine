import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import official_protocol_bootstrap as bootstrap


class BootstrapTests(unittest.TestCase):
    def test_private_cann_does_not_leak_image_python_or_torch_into_protocol_venv(self):
        with tempfile.TemporaryDirectory() as d:
            sdk = Path(d) / 'sdk'
            sdk.mkdir()
            (sdk / 'installed-files.json').write_bytes(b'{}')
            pin = hashlib.sha256(b'{}').hexdigest()
            image = 'sha256:4ee78def8f33d59d48f116d1dfa793332c23c99ecab4f0d7dd5cd62d0fb4e6c1'
            (sdk / 'installation.json').write_text(json.dumps(dict(image_manifest=image, installed_files_sha256=pin)))
            env = dict(PYTHONHOME='/image/python', PYTHONPATH='/image/packages', VLLMHUST_NO_SITE='1',
                       ASCEND_HOME_PATH='/private/cann', LD_LIBRARY_PATH='/private/cann/lib64:/private/python/lib:/image/torch/lib:/image/torch_npu/lib')
            with patch('official_image.read_lock', return_value=dict(manifest_digest=image)), \
                 patch('official_image.environment', return_value=(Path('/private/python/bin/python3'), env)):
                actual, identity = bootstrap.cann_environment(sdk)
            self.assertNotIn('PYTHONHOME', actual)
            self.assertNotIn('PYTHONPATH', actual)
            self.assertEqual(actual['LD_LIBRARY_PATH'], '/private/cann/lib64:/private/python/lib')
            self.assertEqual(identity['version'], '9.1.0')
            self.assertFalse(identity['host_driver_modified'])
            (sdk / 'installed-files.json').write_bytes(b'changed')
            with patch('official_image.read_lock', return_value=dict(manifest_digest=image)):
                with self.assertRaises(ValueError):
                    bootstrap.cann_environment(sdk)


if __name__ == '__main__':
    unittest.main()
