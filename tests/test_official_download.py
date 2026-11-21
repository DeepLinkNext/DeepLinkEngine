import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import official_baselines
import official_download


class Downloads(unittest.TestCase):
    def test_download_configuration_needs_no_model_or_device(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'download.json'
            p.write_text(json.dumps(dict(cache=str(Path(d)/'cache'), out=str(Path(d)/'out'), versions=['0.23.0'])))
            with patch('official_download.run', return_value=0) as download, \
                 patch('official_baselines.platform.system', side_effect=AssertionError('Download must not check/prepare model runtime')):
                self.assertEqual(official_baselines.main(['--config',str(p),'--download-only','--artifact-digest','sha256:'+'a'*64]),0)
            self.assertEqual(download.call_count,1)
            with self.assertRaises(ValueError):
                official_baselines.configuration(p)

    def test_unlisted_digest_is_refused_before_any_download_or_output(self):
        with tempfile.TemporaryDirectory() as d:
            config=dict(versions=['0.23.0'],cache=str(Path(d)/'cache'),out=str(Path(d)/'out'))
            with patch('official_download.artifacts', return_value={}), patch('baseline_setup.fetch') as fetch:
                with self.assertRaises(ValueError):
                    official_download.run(config, artifact='sha256:'+'a'*64)
                fetch.assert_not_called()
                self.assertFalse(Path(config['out']).exists())

    def test_download_lock_excludes_a_second_process(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            code="""import sys,time,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from baseline_setup import download_lock
r=Path(sys.argv[2]);name=sys.argv[3]
with (r/'download.lock').open('a+b') as f, download_lock(f):
 (r/(name+'-begin')).write_text(str(time.monotonic()))
 time.sleep(.3)
 (r/(name+'-end')).write_text(str(time.monotonic()))
"""
            first=subprocess.Popen([sys.executable,'-I','-B','-c',code,str(ROOT),str(root),'first'])
            deadline=time.monotonic()+10
            while not (root/'first-begin').exists() and time.monotonic()<deadline:
                time.sleep(.01)
            self.assertTrue((root/'first-begin').exists())
            second=subprocess.Popen([sys.executable,'-I','-B','-c',code,str(ROOT),str(root),'second'])
            self.assertEqual(first.wait(timeout=10),0)
            self.assertEqual(second.wait(timeout=10),0)
            self.assertGreaterEqual(float((root/'second-begin').read_text()),float((root/'first-end').read_text()))


if __name__=='__main__':unittest.main()
