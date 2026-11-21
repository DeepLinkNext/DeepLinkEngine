import contextlib
import hashlib
import io
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import official_protocol_download as download


class ProtocolDownload(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name);self.config=dict(cache=str(self.root/'cache'),out=str(self.root/'out'))
        self.raw=b'locked CPU-only fixture';self.pin=hashlib.sha256(self.raw).hexdigest();self.identity='sha256:'+self.pin
        self.row=dict(filename='fixture.part-001',sha256=self.pin,bytes=len(self.raw),url='https://example.invalid/locked-fixture',cache_subdirectory='b0-parts',role='original018-protocol-dependency')

    def invoke(self):
        with patch.object(download,'artifacts',return_value={self.identity:self.row}),contextlib.redirect_stdout(io.StringIO()):
            return download.run(self.config,artifact=self.identity)

    def test_pinned_public_inventory_retains_original_protocol_and_private_environment_identities(self):
        rows=download.artifacts()
        self.assertGreater(len(rows),20)
        self.assertIn('original018-protocol-dependency',{r['role'] for r in rows.values()})
        self.assertIn('common-protocol-environment',{r['role'] for r in rows.values()})
        for identity,row in rows.items():
            self.assertEqual(identity,'sha256:'+row['sha256'])
            self.assertIn(row['cache_subdirectory'],('b0-parts','wheels'))

    def test_one_locked_asset_is_prefetched_without_installer_or_model(self):
        with patch('urllib.request.urlopen',side_effect=lambda *a,**k:io.BytesIO(self.raw)):
            self.assertEqual(self.invoke(),0)
        target=self.root/'cache/b0-parts/fixture.part-001';self.assertEqual(target.read_bytes(),self.raw)
        result=json.loads((self.root/'out/results.json').read_bytes())
        self.assertEqual(result['status'],'completed');self.assertEqual(result['files'][0]['sha256'],self.pin)
        for field in ('model_execution_authorized','extraction_performed','installation_performed'):
            self.assertIs(result[field],False)

    def test_unknown_digest_is_rejected_before_output_or_cache(self):
        with self.assertRaisesRegex(ValueError,'not a pinned'):
            download.run(self.config,artifact='sha256:'+'0'*64)
        self.assertFalse((self.root/'cache').exists());self.assertFalse((self.root/'out').exists())

    def test_matching_local_asset_is_verified_without_redownload(self):
        target=self.root/'cache/b0-parts/fixture.part-001';target.parent.mkdir(parents=True);target.write_bytes(self.raw)
        with patch('urllib.request.urlopen',side_effect=AssertionError('Verified cache must not redownload')):
            self.assertEqual(self.invoke(),0)

    def test_wrong_remote_bytes_leave_failed_result_without_admitted_asset(self):
        with patch('urllib.request.urlopen',side_effect=lambda *a,**k:io.BytesIO(b'x'*len(self.raw))):
            with self.assertRaisesRegex(ValueError,'SHA-256'):self.invoke()
        self.assertFalse((self.root/'cache/b0-parts/fixture.part-001').exists())
        self.assertEqual(json.loads((self.root/'out/results.json').read_bytes())['status'],'failed')

    def test_streamed_response_uses_available_bytes_without_waiting_for_a_large_buffer(self):
        class Stream(io.BytesIO):
            def read(self,*args):raise AssertionError('Do not wait for a complete large read')
            def read1(self,size):return super().read(size)
        with patch('urllib.request.urlopen',side_effect=lambda *a,**k:Stream(self.raw)):
            self.assertEqual(self.invoke(),0)

    def test_continued_trickle_is_bounded_and_never_admitted(self):
        with patch('urllib.request.urlopen',side_effect=lambda *a,**k:io.BytesIO(self.raw[:1])), \
             patch('environment_setup.time.monotonic',side_effect=itertools.count(0,46)):
            with self.assertRaisesRegex(OSError,'Dependency download failed'):self.invoke()
        self.assertFalse((self.root/'cache/b0-parts/fixture.part-001').exists())
        journal=json.loads((self.root/'cache/b0-parts/fixture.part-001.download.json').read_bytes())
        self.assertIn('below 1 KiB/s',journal['attempts'][0]['error'])


if __name__=='__main__':unittest.main()
