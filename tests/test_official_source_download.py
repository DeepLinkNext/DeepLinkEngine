import hashlib
import http.server
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from official import enable_module_imports

enable_module_imports()
import official_baselines as public
import official_source_candidate as source
import official_source_download as download


class SourceDownloadTests(unittest.TestCase):
    def row(self,body,url):
        return dict(filename='cpu-build-fixture.whl',bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),
            url=url,upstream_url=url,stage='03-ascend',package='torch',requirement='torch==2.10.0')

    def test_public_cached_download_does_not_enter_linux_or_sdk(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);row=self.row(b'fixture','https://fixture.invalid/wheel');cache=root/'cache/wheels';cache.mkdir(parents=True);(cache/row['filename']).write_bytes(b'fixture')
            config=root/'evaluation.json';config.write_text(json.dumps(dict(cache=str(root/'cache'),out=str(root/'out'),versions=[source.VERSION])))
            with patch.object(source,'build_dependency_artifacts',return_value={'sha256:'+row['sha256']:row}), \
                 patch.object(public.platform,'machine',side_effect=AssertionError('download must not inspect accelerator architecture')):
                code=public.main(['--config',str(config),'--download-only','--download-source-build-dependencies','--artifact-digest','sha256:'+row['sha256']])
            self.assertEqual(code,0);result=json.loads((root/'out/results.json').read_text())
            self.assertEqual(result['status'],'completed');self.assertTrue(result['files'][0]['cached_before'])
            self.assertFalse(result['model_execution_authorized']);self.assertFalse(result['extraction_performed']);self.assertFalse(result['installation_performed'])

    @unittest.skipUnless(shutil.which('curl'),'native curl unavailable')
    def test_native_public_download_resumes_real_http_prefix(self):
        body=b'cpu-build-cache-fixture'*1024
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                offset=int(self.headers.get('Range','bytes=0-').split('=')[1].split('-')[0]);payload=body[offset:]
                self.send_response(206 if offset else 200)
                if offset:self.send_header('Content-Range',f'bytes {offset}-{len(body)-1}/{len(body)}')
                self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
            def log_message(self,*args):pass
        server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);url=f'http://127.0.0.1:{server.server_port}/fixture.whl';row=self.row(body,url);cache=root/'cache/wheels';cache.mkdir(parents=True);(cache/(row['filename']+'.part')).write_bytes(body[:123])
                config=dict(cache=str(root/'cache'),out=str(root/'out'),versions=[source.VERSION])
                with patch.object(source,'build_dependency_artifacts',return_value={'sha256:'+row['sha256']:row}):self.assertEqual(download.run(config,artifact='sha256:'+row['sha256'],transport='curl'),0)
                self.assertEqual((cache/row['filename']).read_bytes(),body)
                record=json.loads((root/'out/results.json').read_text());self.assertEqual(record['files'][0]['isolated_stage'],'03-ascend');self.assertIsNotNone(record['files'][0]['transport_provenance'])
        finally:server.shutdown();server.server_close();thread.join(timeout=5)

    def test_unknown_digest_and_two_gib_margin_refuse_before_fetch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);row=self.row(b'fixture','https://fixture.invalid/wheel');config=dict(cache=str(root/'cache'),out=str(root/'out'),versions=[source.VERSION])
            with patch.object(source,'build_dependency_artifacts',return_value={'sha256:'+row['sha256']:row}),patch.object(source,'source_archive') as fetch:
                with self.assertRaisesRegex(ValueError,'not a pinned'):download.run(config,artifact='sha256:'+'0'*64)
                with patch.object(shutil,'disk_usage',return_value=SimpleNamespace(free=2*1024**3+len(b'fixture')-1)):
                    with self.assertRaisesRegex(OSError,'2 GiB'):download.run(config,artifact='sha256:'+row['sha256'])
                fetch.assert_not_called();self.assertFalse((root/'out').exists())

    @unittest.skipUnless(shutil.which('curl'),'native curl unavailable')
    def test_bad_full_size_partial_is_preserved_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);row=self.row(b'fixture','https://fixture.invalid/wheel');cache=root/'cache/wheels';cache.mkdir(parents=True);part=cache/(row['filename']+'.part');part.write_bytes(b'badbyte')
            config=dict(cache=str(root/'cache'),out=str(root/'out'),versions=[source.VERSION])
            with patch.object(source,'build_dependency_artifacts',return_value={'sha256:'+row['sha256']:row}),patch.object(source,'source_archive') as fetch:
                with self.assertRaisesRegex(ValueError,'wrong whole SHA'):download.run(config,artifact='sha256:'+row['sha256'],transport='curl')
                fetch.assert_not_called()
            self.assertEqual(part.read_bytes(),b'badbyte');self.assertEqual(json.loads((root/'out/results.json').read_text())['status'],'failed')


if __name__=='__main__':unittest.main()
