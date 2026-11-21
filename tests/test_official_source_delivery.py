"""Actual pip cache API and resolver tests; only loopback fixture wheels."""
from pathlib import Path
import base64
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import zipfile
from urllib.parse import quote,unquote
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import official_source_delivery as delivery


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name);self.cache=self.root/'cache';(self.cache/'wheels').mkdir(parents=True)
        self.requests=[];self.rows=[];self.index_urls={}
        owner=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                owner.requests.append(self.path)
                if self.path=='/simple/delivery-probe/':
                    raw=owner.index_body();kind='text/html'
                elif self.path.startswith('/mirror/'):
                    self.send_response(301);self.send_header('Location','/packages/'+unquote(self.path.rsplit('/',1)[-1]));self.send_header('Content-Length','0');self.end_headers();return
                elif self.path.startswith('/packages/'):
                    file=owner.cache/'wheels'/self.path.rsplit('/',1)[-1]
                    if not file.is_file():self.send_error(404);return
                    raw=file.read_bytes();kind='application/octet-stream'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=self.server.serve_forever,daemon=True);thread.start()
        self.addCleanup(self.server.server_close);self.addCleanup(self.server.shutdown)
        self.origin='http://127.0.0.1:'+str(self.server.server_port)
        for version in ('1.0','2.0'):self.rows.append(self.wheel(version))
        self.pip_caches=[self.cache/'source-build-downloads/metadata',self.cache/'source-build-downloads/build/pip']

    def index_body(self):
        return ''.join('<a href="'+self.index_urls.get(r['filename'],'/packages/'+r['filename'])+'#sha256='+r['sha256']+'">'+r['filename']+'</a>' for r in self.rows).encode()

    def alias(self,row):
        url=self.origin+'/mirror/'+quote(row['filename'],safe='')
        self.index_urls[row['filename']]=url
        index=self.root/'actual-index.html';index.write_bytes(self.index_body())
        return dict(url=url,attestation=dict(index=self.origin+'/simple/delivery-probe/',index_path=str(index),index_sha256=delivery.sha(index),filename=row['filename'],advertised_whole_sha256=row['sha256'],matching_urls=[url]))

    def wheel(self,version):
        name='delivery_probe';dist=name+'-'+version+'.dist-info';filename=name+'-'+version+'-py3-none-any.whl'
        path=self.cache/'wheels'/filename
        files={name+'/__init__.py':('VERSION='+repr(version)+'\n').encode(),dist+'/METADATA':('Metadata-Version: 2.1\nName: delivery-probe\nVersion: '+version+'\n').encode(),dist+'/WHEEL':b'Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n'}
        records=[]
        with zipfile.ZipFile(path,'w') as archive:
            for key,raw in files.items():
                archive.writestr(key,raw);records.append([key,'sha256='+base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('='),str(len(raw))])
            records.append([dist+'/RECORD','','']);text=io.StringIO();csv.writer(text,lineterminator='\n').writerows(records);archive.writestr(dist+'/RECORD',text.getvalue())
        return dict(filename=filename,package='delivery-probe',version=version,requirement='delivery-probe=='+version,bytes=path.stat().st_size,sha256=delivery.sha(path),url=self.origin+'/packages/'+filename)

    def request(self,rows=None):
        return dict(pip_version='26.2',cache_root=str(self.cache),pip_caches=list(map(str,self.pip_caches)),artifacts=self.rows if rows is None else rows)

    def resolve(self,version,cache):
        import pip
        env={k:v for k,v in os.environ.items() if not k.startswith('PIP_')}
        env.update(PIP_CONFIG_FILE=os.devnull,PIP_CACHE_DIR=str(cache),PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(Path(pip.__file__).resolve().parent.parent))
        report=self.root/('report-'+version+'-'+str(len(self.requests))+'.json')
        command=[sys.executable,'-B','-S','-m','pip','install','--dry-run','--ignore-installed','--no-deps','--disable-pip-version-check','--index-url',self.origin+'/simple/','--find-links',str(self.cache/'wheels'),'--report',str(report),'delivery-probe=='+version]
        result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        item=json.loads(report.read_bytes())['install'][0]
        self.assertEqual(item['metadata']['version'],version)
        return item

    def test_real_resolver_reuses_original_binary_url_without_body_get_for_both_versions(self):
        # The unprimed control reproduces the actual redundant download.
        self.resolve('1.0',self.root/'unprimed')
        self.assertIn('/packages/'+self.rows[0]['filename'],self.requests)
        proof=delivery.prime(self.request())
        self.assertEqual(len(proof['entries']),4)
        self.assertFalse(proof['local_build_claimed']);self.assertFalse(proof['http_response_or_publisher_headers_fabricated'])
        before=len(self.requests)
        for cache in self.pip_caches:
            for row in self.rows:
                item=self.resolve(row['version'],cache)
                self.assertEqual(item['download_info']['url'],row['url'])
                self.assertEqual(item['download_info']['archive_info']['hashes']['sha256'],row['sha256'])
        self.assertTrue(self.requests[before:])
        self.assertTrue(all(path=='/simple/delivery-probe/' for path in self.requests[before:]))
        reused=delivery.prime(self.request());self.assertTrue(all(r['reused'] for r in reused['entries']))

    def test_bad_canonical_hash_rejected_before_any_native_entry(self):
        path=self.cache/'wheels'/self.rows[0]['filename'];path.write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError,'whole SHA'):delivery.prime(self.request())
        self.assertFalse(self.pip_caches[0].exists());self.assertEqual(path.read_bytes(),b'bad')

    def test_wrong_declared_version_and_wrong_requirement_rejected(self):
        for changes in ({'version':'9.0'},{'requirement':'delivery-probe==9.0'}):
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError,'version'):delivery.prime(self.request([dict(self.rows[0],**changes)]))
        self.assertFalse(self.pip_caches[0].exists())

    def test_unknown_pip_identity_rejected(self):
        request=self.request();request['pip_version']='0.0'
        with self.assertRaisesRegex(ValueError,'pinned actual pip'):delivery.prime(request)
        self.assertFalse(self.pip_caches[0].exists())

    def test_bad_native_hash_and_origin_preserved(self):
        proof=delivery.prime(self.request());first=Path(proof['entries'][0]['cache_path'])
        origin=first/'origin.json';origin.write_text('{"url":"https://wrong.invalid/file.whl"}')
        old=origin.read_bytes()
        with self.assertRaisesRegex(ValueError,'origin differs'):delivery.prime(self.request())
        self.assertEqual(origin.read_bytes(),old)
        shutil.rmtree(first);delivery.prime(self.request())
        wheel=first/self.rows[0]['filename'];wheel.write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError,'cache differs'):delivery.prime(self.request())
        self.assertEqual(wheel.read_bytes(),b'bad')

    def test_source_archive_is_never_given_a_wheel_or_build_origin(self):
        result=delivery.prime(self.request([dict(filename='original.tar.gz')]))
        self.assertEqual(result['entries'],[]);self.assertEqual(len(result['skipped']),1)
        self.assertFalse(any(p.name=='origin.json' for p in self.cache.rglob('origin.json')))

    def test_canonical_wheel_cannot_supply_a_source_origin(self):
        with self.assertRaisesRegex(ValueError,'source/build origin'):delivery.prime(self.request([dict(self.rows[0],url='https://example.com/source.tar.gz')]))
        self.assertFalse(self.pip_caches[0].exists())

    def test_real_pip_alias_key_preserves_canonical_origin_and_encoded_redirect_link(self):
        row=self.wheel('3.0+cpu');self.rows=[row];alias=self.alias(row)
        self.assertIn('%2B',alias['url'])
        # This is the same original transport/index contract used before the
        # recursive Graph preflight, without touching production network state.
        import official_build_assets as assets
        proof=dict(index=alias['attestation']['index'],index_sha256=alias['attestation']['index_sha256'],matching_urls=[alias['url']])
        with patch.object(assets,'transports',return_value=([alias['url']],[proof])):
            rows=delivery.primary_aliases([row],self.root/'primary-index-receipts')
        result=delivery.prime(self.request(rows));self.assertEqual(len(result['entries']),4)
        before=len(self.requests)
        for cache in self.pip_caches:
            item=self.resolve('3.0+cpu',cache)
            self.assertEqual(item['download_info']['url'],row['url'])
        self.assertTrue(all(p=='/simple/delivery-probe/' for p in self.requests[before:]))
        entries=[r for r in result['entries'] if r['native_cache_key_url']==alias['url']]
        self.assertEqual(len(entries),2)
        for entry in entries:
            origin=json.loads((Path(entry['cache_path'])/'origin.json').read_bytes())
            self.assertEqual(origin['url'],row['url']);self.assertFalse(entry['observed_http_fetch'])

    def test_alias_rejects_wrong_index_bytes_and_unadvertised_or_mismatched_digest(self):
        row=self.rows[0];alias=self.alias(row)
        original=Path(alias['attestation']['index_path']).read_bytes()
        for case in ['wrong-index-sha','wrong-advertised-hash','wrong-html-hash']:
            with self.subTest(case=case):
                current=json.loads(json.dumps(alias));index=Path(current['attestation']['index_path']);index.write_bytes(original)
                if case=='wrong-index-sha':current['attestation']['index_sha256']='a'*64
                elif case=='wrong-advertised-hash':current['attestation']['advertised_whole_sha256']='0'*64
                else:
                    index.write_bytes(original.replace(row['sha256'].encode(),b'0'*64));current['attestation']['index_sha256']=delivery.sha(index)
                with self.assertRaisesRegex(ValueError,'index|whole SHA'):delivery.prime(self.request([dict(row,cache_aliases=[current])]))
                self.assertFalse(self.pip_caches[0].exists())


if __name__=='__main__':unittest.main()
