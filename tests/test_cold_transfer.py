import hashlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import baseline_setup


class ColdTransfer(unittest.TestCase):
    def test_trickling_python_connection_falls_back_to_hashed_curl(self):
        payload = b'verified public image bytes'
        class Slow:
            status = 200
            headers = {}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, size): raise AssertionError('Blocking read must not hide trickling traffic')
            def read1(self, size): return payload[:1]
        def curl(args, **kwargs):
            Path(args[args.index('--output') + 1]).write_bytes(payload)
            return types.SimpleNamespace(returncode=0, stderr='')
        with tempfile.TemporaryDirectory() as d, \
             patch('baseline_setup.urllib.request.urlopen', side_effect=lambda *a, **k: Slow()) as python, \
             patch('baseline_setup.time.monotonic', side_effect=[0,46,46,92,92,138]), \
             patch('baseline_setup.time.sleep'), patch('baseline_setup.shutil.which', return_value='/usr/bin/curl'), \
             patch('baseline_setup.subprocess.run', side_effect=curl) as fallback:
            pin = 'sha256:' + hashlib.sha256(payload).hexdigest()
            target = baseline_setup.fetch(d, 'https://example.invalid/', 'blobs', pin, len(payload))
            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(python.call_count, 3)
            self.assertEqual(fallback.call_count, 1)


if __name__ == '__main__': unittest.main()
