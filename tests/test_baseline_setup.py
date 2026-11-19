"""Exercise bounded baseline download recovery and immutable byte checks."""
from pathlib import Path
import hashlib,io,sys,tempfile,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'benchmark/one-click-kit/scripts')]
import baseline_setup as baseline
import one_click
class Response(io.BytesIO):
    def __init__(self,body,status=200,headers=None):
        super().__init__(body);self.status=status;self.headers=headers or {}
class BaselineDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        disabled_curl=patch.object(baseline.shutil,'which',return_value=None);disabled_curl.start();self.addCleanup(disabled_curl.stop)
        self.cache=Path(self.temp.name);self.body=b'abcdef';self.identity='sha256:'+hashlib.sha256(self.body).hexdigest()
    def call(self):return baseline.fetch(self.cache,'https://example.invalid/','blobs',self.identity,len(self.body))
    def test_short_download_resumes_at_last_complete_byte(self):
        with patch.object(baseline.urllib.request,'urlopen',side_effect=[Response(b'ab'),Response(b'cdef',206,{'Content-Range':'bytes 2-5/6'})]) as network,patch.object(baseline.time,'sleep'):
            target=self.call()
        self.assertEqual(target.read_bytes(),self.body)
        self.assertEqual(network.call_args_list[1].args[0].get_header('Range'),'bytes=2-')
    def test_server_ignoring_range_restarts_without_duplicating_bytes(self):
        with patch.object(baseline.urllib.request,'urlopen',side_effect=[Response(b'ab'),Response(self.body)]),patch.object(baseline.time,'sleep'):
            self.assertEqual(self.call().read_bytes(),self.body)
    def test_wrong_range_is_never_published(self):
        with patch.object(baseline.urllib.request,'urlopen',side_effect=[Response(b'ab'),Response(b'def',206,{'Content-Range':'bytes 3-5/6'})]),patch.object(baseline.time,'sleep'):
            with self.assertRaisesRegex(ValueError,'range'):self.call()
        self.assertFalse((self.cache/self.identity.replace(':','-')).exists())
    def test_timeouts_are_bounded(self):
        with patch.object(baseline.urllib.request,'urlopen',side_effect=TimeoutError('handshake')) as network,patch.object(baseline.time,'sleep'):
            with self.assertRaisesRegex(OSError,'bounded retries'):self.call()
        self.assertEqual(network.call_count,3)
    def test_same_size_wrong_bytes_are_not_accepted(self):
        with patch.object(baseline.urllib.request,'urlopen',return_value=Response(b'xxxxxx')):
            with self.assertRaisesRegex(ValueError,'SHA-256'):self.call()
        self.assertFalse((self.cache/self.identity.replace(':','-')).exists())
    def test_verified_cache_is_usable_offline(self):
        target=self.cache/self.identity.replace(':','-');target.write_bytes(self.body)
        with patch.object(baseline.urllib.request,'urlopen',side_effect=AssertionError('network')):
            self.assertEqual(self.call(),target)
    def test_explicit_baseline_uses_original_admission_without_downloading(self):
        args=['--cache',str(self.cache),'--baseline-deployment','/user/sdk']
        with patch.object(baseline,'prepare') as prepare:
            self.assertEqual(one_click.prepare_baseline(args),args)
        prepare.assert_not_called()
    def test_dangling_cache_symlink_is_rejected(self):
        (self.cache/self.identity.replace(':','-')).symlink_to(self.cache/'missing')
        with self.assertRaisesRegex(ValueError,'symlink'):self.call()
    def test_curl_transport_fallback_keeps_original_hash_contract(self):
        def transport(command,**kwargs):
            Path(command[command.index('--output')+1]).write_bytes(self.body)
            return type('Result',(),{'returncode':0,'stderr':''})()
        with patch.object(baseline.urllib.request,'urlopen',side_effect=TimeoutError('TLS')),patch.object(baseline.time,'sleep'),patch.object(baseline.shutil,'which',return_value='/curl'),patch.object(baseline.subprocess,'run',side_effect=transport) as curl:
            self.assertEqual(self.call().read_bytes(),self.body)
        self.assertIn('--continue-at',curl.call_args.args[0])
    def test_curl_success_cannot_hide_wrong_payload(self):
        def transport(command,**kwargs):
            Path(command[command.index('--output')+1]).write_bytes(b'xxxxxx')
            return type('Result',(),{'returncode':0,'stderr':''})()
        with patch.object(baseline.urllib.request,'urlopen',side_effect=TimeoutError('TLS')),patch.object(baseline.time,'sleep'),patch.object(baseline.shutil,'which',return_value='/curl'),patch.object(baseline.subprocess,'run',side_effect=transport):
            with self.assertRaisesRegex(ValueError,'SHA-256'):self.call()
        self.assertFalse((self.cache/self.identity.replace(':','-')).exists())
if __name__=='__main__':unittest.main()
