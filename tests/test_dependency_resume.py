"""Production cached_wheel with real local truncated HTTP and range responses."""
import contextlib
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import threading
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from environment_setup import cached_wheel


class Resume(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.cache=Path(self.temp.name)
        self.body=b'abcdef';self.cap=3;self.mode='correct';self.requests=[];fixture=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                header=self.headers.get('Range');fixture.requests.append((self.path,header))
                start=int(re.fullmatch(r'bytes=(\d+)-',header)[1]) if header else 0
                if header and (fixture.mode=='unavailable' or fixture.mode=='api-unavailable' and self.path=='/api'):
                    self.send_error(503);return
                raw=fixture.body[start:start+fixture.cap]
                code=206 if header else 200
                if header and fixture.mode=='ignored':code=200;raw=fixture.body
                if fixture.mode=='corrupt':raw=b'x'*len(fixture.body)
                if fixture.mode=='oversize':raw=fixture.body+b'x'
                length=len(fixture.body) if fixture.mode=='premature-eof' and not header else len(raw)
                self.send_response(code);self.send_header('Content-Length',str(length))
                if code==206 and fixture.mode!='missing-range':
                    first=0 if fixture.mode=='wrong-start' else start
                    total=len(fixture.body)+int(fixture.mode=='wrong-total')
                    self.send_header('Content-Range',f'bytes {first}-{start+len(raw)-1}/{total}')
                self.end_headers();self.wfile.write(raw)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.addCleanup(self.stop)
        self.url=f'http://127.0.0.1:{self.server.server_port}/fixture.whl'

    def stop(self):
        self.server.shutdown();self.server.server_close();self.thread.join()

    def source(self):
        return dict(filename='fixture.whl',bytes=len(self.body),sha256=hashlib.sha256(self.body).hexdigest(),url=self.url)

    def fetch(self):
        with contextlib.redirect_stdout(io.StringIO()):return cached_wheel(self.source(),[],self.cache)

    def journal(self):return json.loads((self.cache/'fixture.whl.download.json').read_bytes())

    def test_truncated_transfer_continues_exact_offset_and_whole_sha(self):
        path=self.fetch();self.assertEqual(path.read_bytes(),self.body)
        self.assertEqual([h for p,h in self.requests],[None,'bytes=3-'])
        j=self.journal();self.assertEqual(j['status'],'completed');self.assertEqual(j['completed_file']['sha256'],self.source()['sha256'])
        self.assertEqual(j['attempts'][0]['partial_bytes'],3);self.assertEqual(j['attempts'][1]['offset'],3)

    def test_multiple_large_capped_responses_advance_without_restarting(self):
        self.body=b'x'*(3*1024**2+7);self.cap=1024**2
        self.assertEqual(self.fetch().read_bytes(),self.body)
        self.assertEqual([h for p,h in self.requests],[None,'bytes=1048576-','bytes=2097152-','bytes=3145728-'])

    def test_premature_eof_with_full_content_length_still_resumes(self):
        self.mode='premature-eof'
        self.assertEqual(self.fetch().read_bytes(),self.body)
        self.assertEqual([h for p,h in self.requests],[None,'bytes=3-'])

    def test_fallback_uses_only_fixed_urls_and_retains_the_same_prefix(self):
        self.mode='api-unavailable';source=self.source();source['api_url']=self.url.rsplit('/',1)[0]+'/api'
        with contextlib.redirect_stdout(io.StringIO()):path=cached_wheel(source,[],self.cache)
        self.assertEqual(path.read_bytes(),self.body)
        self.assertEqual(self.requests,[('/api',None),('/fixture.whl','bytes=3-')])
        self.assertEqual([r['url'] for r in self.journal()['attempts']],[source['api_url'],source['url']])

    def test_failed_invocation_retains_prefix_and_next_invocation_resumes(self):
        self.mode='unavailable'
        with self.assertRaises(OSError):self.fetch()
        self.assertEqual((self.cache/'fixture.whl.download.partial').read_bytes(),b'abc')
        old=self.journal();self.assertEqual(old['status'],'failed')
        self.mode='correct';self.requests=[]
        self.assertEqual(self.fetch().read_bytes(),self.body)
        self.assertEqual([h for p,h in self.requests],['bytes=3-'])
        self.assertGreater(len(self.journal()['attempts']),len(old['attempts']))

    def test_ignored_missing_or_incorrect_ranges_never_append_or_publish(self):
        for mode in ('ignored','missing-range','wrong-start','wrong-total'):
            with self.subTest(mode=mode):
                self.mode=mode
                with self.assertRaises(OSError):self.fetch()
                self.assertFalse((self.cache/'fixture.whl').exists())
                self.assertEqual((self.cache/'fixture.whl.download.partial').read_bytes(),b'abc')
                self.assertIn('exact206',self.journal()['attempts'][-1]['error'])

    def test_wrong_whole_sha_is_retained_and_never_admitted(self):
        self.mode='corrupt'
        with self.assertRaisesRegex(ValueError,'SHA-256'):self.fetch()
        self.assertFalse((self.cache/'fixture.whl').exists());self.assertEqual(self.journal()['status'],'failed')
        self.assertEqual((self.cache/'fixture.whl.download.partial').read_bytes(),b'xxxxxx')

    def test_oversized_response_never_writes_bytes_beyond_the_fixed_size(self):
        self.mode='oversize'
        with self.assertRaisesRegex(ValueError,'exceeds its fixed size'):self.fetch()
        self.assertFalse((self.cache/'fixture.whl').exists())
        self.assertEqual((self.cache/'fixture.whl.download.partial').stat().st_size,0)

    def test_modified_partial_is_refused_before_network(self):
        self.mode='unavailable'
        with self.assertRaises(OSError):self.fetch()
        (self.cache/'fixture.whl.download.partial').write_bytes(b'Abc');self.requests=[]
        with self.assertRaisesRegex(ValueError,'changed outside'):self.fetch()
        self.assertFalse(self.requests)

    def test_unowned_partial_is_refused_without_moving_or_fetching(self):
        part=self.cache/'fixture.whl.download.partial';part.write_bytes(b'abc')
        with self.assertRaisesRegex(ValueError,'Unowned'):self.fetch()
        self.assertEqual(part.read_bytes(),b'abc');self.assertFalse(self.requests)

    def test_pinned_source_change_refuses_existing_partial(self):
        self.mode='unavailable'
        with self.assertRaises(OSError):self.fetch()
        self.body=b'ABCDEF';self.requests=[]
        with self.assertRaisesRegex(ValueError,'different pinned source'):self.fetch()
        self.assertFalse(self.requests)


if __name__=='__main__':unittest.main()
